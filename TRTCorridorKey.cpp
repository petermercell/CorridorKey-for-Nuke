/*
 * TRTCorridorKey - TensorRT CorridorKey Inference Node for Nuke
 *
 * Model details (from ONNX export / trtexec):
 *   Input:  "input"  float32[1,4,2048,2048]  (NCHW: ImageNet-normed RGB + raw alpha hint)
 *   Output: "alpha"  float32[1,1,2048,2048]  (linear alpha, post-sigmoid)
 *   Output: "fg"     float32[1,3,2048,2048]  (sRGB straight FG, post-sigmoid)
 *
 * Two Nuke inputs:
 *   Input 0: RGB plate (green screen footage)
 *   Input 1: Alpha hint (coarse matte — single channel or RGB where R is used)
 *
 * Output: RGBA — FG from model in RGB, alpha from model in A
 *
 * Built for: TensorRT 10.15.1 (enqueueV3 + named tensors)
 *            Nuke 17.0 NDK
 *            CUDA 13.1
 *
 * Author: Peter Mercell
 * Website: petermercell.com
 *
 * CorridorKey model by Niko Pueringer / Corridor Digital
 * https://github.com/nikopueringer/CorridorKey
 * License: CC BY-NC-SA 4.0
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"
#include "DDImage/Interest.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mutex>

using namespace DD::Image;

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TRTCKLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRTCorridorKey] " << msg << std::endl;
    }
};

static TRTCKLogger gLogger;

// ---------------------------------------------------------------------------
// Tensor names (from ONNX export)
// ---------------------------------------------------------------------------
static const char* kInputTensorName  = "input";
static const char* kAlphaTensorName  = "alpha";
static const char* kFGTensorName     = "fg";

// ---------------------------------------------------------------------------
// Resolution presets
// ---------------------------------------------------------------------------
static const char* const kResolutionNames[] = {
    "512x512",
    "768x768",
    "1024x1024",
    "2048x2048",
    "Custom",
    nullptr
};

// ---------------------------------------------------------------------------
// Output mode presets
// ---------------------------------------------------------------------------
static const char* const kOutputModeNames[] = {
    "RGBA (FG + Alpha)",
    "Alpha Only",
    "FG Only",
    nullptr
};

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            error("CUDA error: %s at %s:%d", cudaGetErrorString(err),           \
                  __FILE__, __LINE__);                                           \
            return;                                                             \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Bilinear resize (CPU, planar)
// ---------------------------------------------------------------------------
static void bilinearResize(const float* src, int srcW, int srcH,
                           float* dst, int dstW, int dstH,
                           int channels)
{
    for (int c = 0; c < channels; ++c)
    {
        const float* srcC = src + c * srcW * srcH;
        float* dstC       = dst + c * dstW * dstH;

        for (int dy = 0; dy < dstH; ++dy)
        {
            float sy = (dy + 0.5f) * srcH / (float)dstH - 0.5f;
            int y0   = (int)std::floor(sy);
            int y1   = y0 + 1;
            float fy = sy - y0;

            y0 = std::max(0, std::min(y0, srcH - 1));
            y1 = std::max(0, std::min(y1, srcH - 1));

            for (int dx = 0; dx < dstW; ++dx)
            {
                float sx = (dx + 0.5f) * srcW / (float)dstW - 0.5f;
                int x0   = (int)std::floor(sx);
                int x1   = x0 + 1;
                float fx = sx - x0;

                x0 = std::max(0, std::min(x0, srcW - 1));
                x1 = std::max(0, std::min(x1, srcW - 1));

                float v00 = srcC[y0 * srcW + x0];
                float v10 = srcC[y0 * srcW + x1];
                float v01 = srcC[y1 * srcW + x0];
                float v11 = srcC[y1 * srcW + x1];

                dstC[dy * dstW + dx] =
                    (1 - fy) * ((1 - fx) * v00 + fx * v10)
                  +      fy  * ((1 - fx) * v01 + fx * v11);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TRTCorridorKey node
// ---------------------------------------------------------------------------
class TRTCorridorKey : public Iop
{

public:

    TRTCorridorKey(Node* node);
    ~TRTCorridorKey() override;

    int minimum_inputs() const override { return 2; }
    int maximum_inputs() const override { return 2; }

    const char* input_label(int n, char*) const override
    {
        switch (n) {
            case 0: return "plate";
            case 1: return "mask";
            default: return "";
        }
    }

    const char* Class() const override { return description.name; }
    const char* node_help() const override
    {
        return "TensorRT CorridorKey inference.\n\n"
               "AI green screen keyer that produces physically\n"
               "accurate unmixed foreground color and alpha.\n\n"
               "Input 0 (plate): RGB green screen footage\n"
               "Input 1 (mask):  Coarse alpha hint\n\n"
               "Output: RGBA — FG color (sRGB) + linear alpha\n\n"
               "Engine tensors:\n"
               "  input:  [1,4,2048,2048] (RGB norm + mask)\n"
               "  alpha:  [1,1,2048,2048] (post-sigmoid)\n"
               "  fg:     [1,3,2048,2048] (post-sigmoid)\n\n"
               "Sigmoid already applied in model — no extra\n"
               "activation needed.\n\n"
               "TRTCorridorKey by Peter Mercell, 2026\n"
               "petermercell.com\n\n"
               "CorridorKey model by Niko Pueringer\n"
               "https://github.com/nikopueringer/CorridorKey\n"
               "License: CC BY-NC-SA 4.0";
    }

    void knobs(Knob_Callback f) override;
    int knob_changed(Knob* k) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t,
                  ChannelMask channels, int count) override;
    void _open() override;
    void _close() override;
    void engine(int y, int x, int r,
                ChannelMask channels, Row& row) override;

    static const Iop::Description description;

private:

    // --- knobs ---
    const char* enginePath_;
    int         resolutionEnum_;
    int         modelW_;
    int         modelH_;
    int         outputMode_;       // 0=RGBA, 1=alpha only, 2=FG only
    bool        invertMatte_;
    int         gpuDevice_;

    // --- frame geometry ---
    int frameW_;
    int frameH_;
    int frameX_;
    int frameY_;

    // --- CPU buffers ---
    // Input: plate RGB (3ch) + mask (1ch) = 4 channels total
    std::vector<float> cpuPlateIn_;    // 3 * frameW * frameH (planar CHW)
    std::vector<float> cpuMaskIn_;     // 1 * frameW * frameH (planar)
    std::vector<float> modelInput_;    // 4 * modelW * modelH (planar NCHW)

    // Output: alpha (1ch) + FG (3ch)
    std::vector<float> modelAlphaOut_; // 1 * modelW * modelH
    std::vector<float> modelFGOut_;    // 3 * modelW * modelH

    // Resized back to frame resolution
    std::vector<float> cpuAlphaOut_;   // frameW * frameH
    std::vector<float> cpuFGOut_;      // 3 * frameW * frameH (planar CHW)

    // --- CUDA ---
    float*       d_input_;
    float*       d_alpha_;
    float*       d_fg_;
    cudaStream_t stream_;

    // --- TensorRT 10.x ---
    nvinfer1::IRuntime*          runtime_;
    nvinfer1::ICudaEngine*       engineTRT_;
    nvinfer1::IExecutionContext*  context_;

    // --- frame-level inference lock ---
    std::mutex inferenceMutex_;
    bool       inferenceRan_;

    // --- engine state ---
    bool engineLoaded_;

    // --- methods ---
    void loadEngine();
    void freeEngine();
    void allocateGPU();
    void freeGPU();
    void fetchAllRows();
    void preprocessFrame();
    void runInference();
    void postprocessOutputs();
    void doFullInference();

    int resolvedModelW() const;
    int resolvedModelH() const;
};

// ---------------------------------------------------------------------------
// Ctor / Dtor
// ---------------------------------------------------------------------------
TRTCorridorKey::TRTCorridorKey(Node* node)
    : Iop(node)
    , enginePath_("")
    , resolutionEnum_(3)       // default 2048x2048
    , modelW_(2048)
    , modelH_(2048)
    , outputMode_(0)
    , invertMatte_(false)
    , gpuDevice_(0)
    , frameW_(0), frameH_(0), frameX_(0), frameY_(0)
    , d_input_(nullptr), d_alpha_(nullptr), d_fg_(nullptr), stream_(nullptr)
    , runtime_(nullptr), engineTRT_(nullptr), context_(nullptr)
    , inferenceRan_(false)
    , engineLoaded_(false)
{
}

TRTCorridorKey::~TRTCorridorKey()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------
void TRTCorridorKey::knobs(Knob_Callback f)
{
    File_knob(f, &enginePath_, "engine_path", "Engine File");
    Tooltip(f, "Path to the TensorRT .engine file.\n"
               "e.g. CorridorKey_v1.0_fp16.engine");

    Divider(f, "Model");

    Enumeration_knob(f, &resolutionEnum_, kResolutionNames,
                     "resolution", "Resolution");
    Tooltip(f, "Model input resolution. Must match the engine.\n"
               "CorridorKey native resolution is 2048x2048.");

    Int_knob(f, &modelW_, "model_w", "Custom W");
    Int_knob(f, &modelH_, "model_h", "Custom H");
    SetFlags(f, Knob::HIDDEN);

    Divider(f, "Output");

    Enumeration_knob(f, &outputMode_, kOutputModeNames,
                     "output_mode", "Output");
    Tooltip(f, "RGBA: FG color in RGB + alpha in A\n"
               "Alpha Only: matte in RGB + A\n"
               "FG Only: FG color in RGB, no alpha");

    Bool_knob(f, &invertMatte_, "invert", "Invert Matte");

    Divider(f, "Advanced");

    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");

    Divider(f, "");
    Text_knob(f, "TRTCorridorKey by Peter Mercell, 2026\n"
                 "petermercell.com\n"
                 "CorridorKey model: github.com/nikopueringer/CorridorKey");
}

int TRTCorridorKey::knob_changed(Knob* k)
{
    if (k->is("resolution"))
    {
        bool custom = (resolutionEnum_ == 4);
        knob("model_w")->visible(custom);
        knob("model_h")->visible(custom);
        return 1;
    }
    return Iop::knob_changed(k);
}

// ---------------------------------------------------------------------------
// Resolution helpers
// ---------------------------------------------------------------------------
int TRTCorridorKey::resolvedModelW() const
{
    switch (resolutionEnum_) {
        case 0: return 512;
        case 1: return 768;
        case 2: return 1024;
        case 3: return 2048;
        default: return modelW_;
    }
}

int TRTCorridorKey::resolvedModelH() const
{
    switch (resolutionEnum_) {
        case 0: return 512;
        case 1: return 768;
        case 2: return 1024;
        case 3: return 2048;
        default: return modelH_;
    }
}

// ---------------------------------------------------------------------------
// Load TensorRT engine (TRT 10.x)
// ---------------------------------------------------------------------------
void TRTCorridorKey::loadEngine()
{
    if (engineLoaded_) return;
    if (!enginePath_ || strlen(enginePath_) == 0)
    {
        error("No TensorRT engine file specified.");
        return;
    }

    std::ifstream file(enginePath_, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        error("Cannot open engine file: %s", enginePath_);
        return;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize))
    {
        error("Failed to read engine file.");
        return;
    }
    file.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_)
    {
        error("Failed to create TensorRT runtime.");
        return;
    }

    engineTRT_ = runtime_->deserializeCudaEngine(buffer.data(), fileSize);
    if (!engineTRT_)
    {
        error("Failed to deserialize TensorRT engine from: %s", enginePath_);
        return;
    }

    // Verify tensor names (TRT 10.x API)
    int nbIO = engineTRT_->getNbIOTensors();
    bool foundInput = false, foundAlpha = false, foundFG = false;

    for (int i = 0; i < nbIO; ++i)
    {
        const char* name = engineTRT_->getIOTensorName(i);
        if (strcmp(name, kInputTensorName) == 0) foundInput = true;
        if (strcmp(name, kAlphaTensorName) == 0) foundAlpha = true;
        if (strcmp(name, kFGTensorName) == 0)    foundFG = true;
    }

    if (!foundInput)
    {
        error("Engine missing input tensor '%s'", kInputTensorName);
        freeEngine();
        return;
    }
    if (!foundAlpha)
    {
        error("Engine missing output tensor '%s'", kAlphaTensorName);
        freeEngine();
        return;
    }
    if (!foundFG)
    {
        error("Engine missing output tensor '%s'", kFGTensorName);
        freeEngine();
        return;
    }

    context_ = engineTRT_->createExecutionContext();
    if (!context_)
    {
        error("Failed to create TensorRT execution context.");
        freeEngine();
        return;
    }

    engineLoaded_ = true;
    std::cerr << "[TRTCorridorKey] Engine loaded: " << enginePath_ << std::endl;
}

void TRTCorridorKey::freeEngine()
{
    if (context_)   { delete context_;   context_   = nullptr; }
    if (engineTRT_) { delete engineTRT_; engineTRT_ = nullptr; }
    if (runtime_)   { delete runtime_;   runtime_   = nullptr; }
    engineLoaded_ = false;
}

// ---------------------------------------------------------------------------
// GPU buffers — 4ch input, 1ch alpha output, 3ch FG output
// ---------------------------------------------------------------------------
void TRTCorridorKey::allocateGPU()
{
    freeGPU();

    int mW = resolvedModelW();
    int mH = resolvedModelH();

    size_t inBytes    = 4 * mW * mH * sizeof(float);
    size_t alphaBytes = 1 * mW * mH * sizeof(float);
    size_t fgBytes    = 3 * mW * mH * sizeof(float);

    cudaSetDevice(gpuDevice_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMalloc(&d_input_, inBytes));
    CUDA_CHECK(cudaMalloc(&d_alpha_, alphaBytes));
    CUDA_CHECK(cudaMalloc(&d_fg_,    fgBytes));

    // TRT 10.x: set tensor addresses by name
    if (!context_->setTensorAddress(kInputTensorName, d_input_))
    {
        error("Failed to set input tensor address");
        return;
    }
    if (!context_->setTensorAddress(kAlphaTensorName, d_alpha_))
    {
        error("Failed to set alpha output tensor address");
        return;
    }
    if (!context_->setTensorAddress(kFGTensorName, d_fg_))
    {
        error("Failed to set fg output tensor address");
        return;
    }
}

void TRTCorridorKey::freeGPU()
{
    if (stream_)  { cudaStreamDestroy(stream_);  stream_  = nullptr; }
    if (d_input_) { cudaFree(d_input_);          d_input_ = nullptr; }
    if (d_alpha_) { cudaFree(d_alpha_);          d_alpha_ = nullptr; }
    if (d_fg_)    { cudaFree(d_fg_);             d_fg_    = nullptr; }
}

// ---------------------------------------------------------------------------
// _open / _close
// ---------------------------------------------------------------------------
void TRTCorridorKey::_open()
{
    loadEngine();
    if (engineLoaded_)
        allocateGPU();
}

void TRTCorridorKey::_close()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// _validate — two inputs, RGBA output
// ---------------------------------------------------------------------------
void TRTCorridorKey::_validate(bool for_real)
{
    // Validate both inputs
    copy_info();                // copies format from input 0
    input(1)->validate(for_real);

    ChannelSet out = info().channels();
    out += Chan_Red;
    out += Chan_Green;
    out += Chan_Blue;
    out += Chan_Alpha;
    set_out_channels(out);
    info_.turn_on(Chan_Alpha);

    const Format& fmt = info().format();
    frameX_ = fmt.x();
    frameY_ = fmt.y();
    frameW_ = fmt.w();
    frameH_ = fmt.h();

    if (for_real)
    {
        int mW = resolvedModelW();
        int mH = resolvedModelH();

        cpuPlateIn_.resize(3 * frameW_ * frameH_, 0.0f);
        cpuMaskIn_.resize(frameW_ * frameH_, 0.0f);
        modelInput_.resize(4 * mW * mH, 0.0f);

        modelAlphaOut_.resize(mW * mH, 0.0f);
        modelFGOut_.resize(3 * mW * mH, 0.0f);

        cpuAlphaOut_.resize(frameW_ * frameH_, 0.0f);
        cpuFGOut_.resize(3 * frameW_ * frameH_, 0.0f);

        inferenceRan_ = false;
    }
}

// ---------------------------------------------------------------------------
// _request — request full frame from both inputs
// ---------------------------------------------------------------------------
void TRTCorridorKey::_request(int x, int y, int r, int t,
                              ChannelMask channels, int count)
{
    // Request RGB from plate (input 0)
    ChannelSet plateNeed(Chan_Red);
    plateNeed += Chan_Green;
    plateNeed += Chan_Blue;
    input(0)->request(frameX_, frameY_,
                      frameX_ + frameW_, frameY_ + frameH_,
                      plateNeed, count);

    // Request mask channel from alpha hint (input 1)
    // We read Red channel — works for both single-channel and RGB masks
    ChannelSet maskNeed(Chan_Red);
    input(1)->request(frameX_, frameY_,
                      frameX_ + frameW_, frameY_ + frameH_,
                      maskNeed, count);
}

// ---------------------------------------------------------------------------
// Fetch all input rows into planar buffers (plate + mask)
// Called once under lock
// ---------------------------------------------------------------------------
void TRTCorridorKey::fetchAllRows()
{
    // Fetch plate (input 0) — RGB
    {
        ChannelSet need(Chan_Red);
        need += Chan_Green;
        need += Chan_Blue;

        float* rPlane = cpuPlateIn_.data();
        float* gPlane = cpuPlateIn_.data() + frameW_ * frameH_;
        float* bPlane = cpuPlateIn_.data() + 2 * frameW_ * frameH_;

        for (int y = frameY_; y < frameY_ + frameH_; ++y)
        {
            Row row(frameX_, frameX_ + frameW_);
            input(0)->get(y, frameX_, frameX_ + frameW_, need, row);

            if (aborted()) return;

            const float* rIn = row[Chan_Red]   + frameX_;
            const float* gIn = row[Chan_Green] + frameX_;
            const float* bIn = row[Chan_Blue]  + frameX_;

            int rowIdx = (y - frameY_) * frameW_;

            for (int i = 0; i < frameW_; ++i)
            {
                rPlane[rowIdx + i] = rIn[i];
                gPlane[rowIdx + i] = gIn[i];
                bPlane[rowIdx + i] = bIn[i];
            }
        }
    }

    // Fetch mask (input 1) — Red channel as alpha hint
    {
        ChannelSet need(Chan_Red);

        float* maskPlane = cpuMaskIn_.data();

        for (int y = frameY_; y < frameY_ + frameH_; ++y)
        {
            Row row(frameX_, frameX_ + frameW_);
            input(1)->get(y, frameX_, frameX_ + frameW_, need, row);

            if (aborted()) return;

            const float* mIn = row[Chan_Red] + frameX_;
            int rowIdx = (y - frameY_) * frameW_;

            for (int i = 0; i < frameW_; ++i)
                maskPlane[rowIdx + i] = mIn[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocess: resize plate + mask to model size, ImageNet normalize RGB,
// assemble 4-channel input [R_norm, G_norm, B_norm, mask_raw]
// ---------------------------------------------------------------------------
void TRTCorridorKey::preprocessFrame()
{
    int mW = resolvedModelW();
    int mH = resolvedModelH();
    int mPixels = mW * mH;

    // Resize plate (3ch) into first 3 planes of modelInput_
    // We use a temp buffer for the resize, then normalize in-place
    std::vector<float> plateResized(3 * mPixels);
    bilinearResize(cpuPlateIn_.data(), frameW_, frameH_,
                   plateResized.data(), mW, mH, 3);

    // ImageNet normalize and copy into modelInput_
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std[3]  = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < 3; ++c)
    {
        const float* src = plateResized.data() + c * mPixels;
        float* dst = modelInput_.data() + c * mPixels;
        for (int i = 0; i < mPixels; ++i)
            dst[i] = (src[i] - mean[c]) / std[c];
    }

    // Resize mask (1ch) into 4th plane of modelInput_
    // The mask goes in as raw [0,1] — no normalization
    float* maskDst = modelInput_.data() + 3 * mPixels;
    bilinearResize(cpuMaskIn_.data(), frameW_, frameH_,
                   maskDst, mW, mH, 1);
}

// ---------------------------------------------------------------------------
// TensorRT inference (TRT 10.x: enqueueV3)
// ---------------------------------------------------------------------------
void TRTCorridorKey::runInference()
{
    if (!engineLoaded_ || !context_) return;

    int mW = resolvedModelW();
    int mH = resolvedModelH();

    size_t inBytes    = 4 * mW * mH * sizeof(float);
    size_t alphaBytes = 1 * mW * mH * sizeof(float);
    size_t fgBytes    = 3 * mW * mH * sizeof(float);

    // H2D: upload input
    cudaMemcpyAsync(d_input_, modelInput_.data(), inBytes,
                    cudaMemcpyHostToDevice, stream_);

    // Execute
    bool ok = context_->enqueueV3(stream_);
    if (!ok)
        std::cerr << "[TRTCorridorKey] enqueueV3 FAILED!" << std::endl;

    // D2H: download both outputs
    cudaMemcpyAsync(modelAlphaOut_.data(), d_alpha_, alphaBytes,
                    cudaMemcpyDeviceToHost, stream_);

    cudaMemcpyAsync(modelFGOut_.data(), d_fg_, fgBytes,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// Postprocess: resize alpha + FG back to frame resolution
// No sigmoid needed — model already applies sigmoid
// ---------------------------------------------------------------------------
void TRTCorridorKey::postprocessOutputs()
{
    int mW = resolvedModelW();
    int mH = resolvedModelH();

    // Resize alpha [1, mW, mH] -> [frameW, frameH]
    bilinearResize(modelAlphaOut_.data(), mW, mH,
                   cpuAlphaOut_.data(), frameW_, frameH_, 1);

    // Resize FG [3, mW, mH] -> [3, frameW, frameH]
    bilinearResize(modelFGOut_.data(), mW, mH,
                   cpuFGOut_.data(), frameW_, frameH_, 3);

    if (invertMatte_)
    {
        int pixels = frameW_ * frameH_;
        for (int i = 0; i < pixels; ++i)
            cpuAlphaOut_[i] = 1.0f - cpuAlphaOut_[i];
    }
}

// ---------------------------------------------------------------------------
// Full inference pipeline (called once per frame under lock)
// ---------------------------------------------------------------------------
void TRTCorridorKey::doFullInference()
{
    fetchAllRows();
    if (aborted()) return;

    preprocessFrame();
    runInference();
    postprocessOutputs();
}

// ---------------------------------------------------------------------------
// engine() — called per scanline by Nuke (multi-threaded)
// ---------------------------------------------------------------------------
void TRTCorridorKey::engine(int y, int x, int r,
                            ChannelMask channels, Row& row)
{
    if (!engineLoaded_)
    {
        input(0)->get(y, x, r, channels, row);
        return;
    }

    // First thread does the full fetch + inference
    {
        std::lock_guard<std::mutex> lock(inferenceMutex_);
        if (!inferenceRan_)
        {
            doFullInference();
            inferenceRan_ = true;
        }
    }

    // Read from pre-computed buffers
    int localY = y - frameY_;
    int rowOffset = localY * frameW_;

    float* rOut = row.writable(Chan_Red);
    float* gOut = row.writable(Chan_Green);
    float* bOut = row.writable(Chan_Blue);

    const float* fgR = cpuFGOut_.data();
    const float* fgG = cpuFGOut_.data() + frameW_ * frameH_;
    const float* fgB = cpuFGOut_.data() + 2 * frameW_ * frameH_;

    switch (outputMode_)
    {
        case 1: // Alpha only — matte in RGB
        {
            for (int i = x; i < r; ++i)
            {
                int localX = i - frameX_;
                float m = cpuAlphaOut_[rowOffset + localX];
                rOut[i] = m;
                gOut[i] = m;
                bOut[i] = m;
            }
            break;
        }
        case 2: // FG only — no alpha
        {
            for (int i = x; i < r; ++i)
            {
                int localX = i - frameX_;
                rOut[i] = fgR[rowOffset + localX];
                gOut[i] = fgG[rowOffset + localX];
                bOut[i] = fgB[rowOffset + localX];
            }
            break;
        }
        default: // RGBA — FG color + alpha
        {
            for (int i = x; i < r; ++i)
            {
                int localX = i - frameX_;
                rOut[i] = fgR[rowOffset + localX];
                gOut[i] = fgG[rowOffset + localX];
                bOut[i] = fgB[rowOffset + localX];
            }
            break;
        }
    }

    // Alpha channel
    if (channels & Mask_Alpha)
    {
        float* aOut = row.writable(Chan_Alpha);
        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            aOut[i] = cpuAlphaOut_[rowOffset + localX];
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
static Iop* build(Node* node)
{
    return new TRTCorridorKey(node);
}

const Iop::Description TRTCorridorKey::description(
    "TRTCorridorKey",
    "AI/TRTCorridorKey",
    build
);
