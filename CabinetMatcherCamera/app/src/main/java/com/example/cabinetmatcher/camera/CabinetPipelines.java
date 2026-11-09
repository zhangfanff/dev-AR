package com.example.cabinetmatcher.camera;

import com.example.cabinetmatcher.camera.core.CabinetPipeline;
import com.example.cabinetmatcher.camera.core.ModelChoice;
import com.example.cabinetmatcher.camera.core.TextEngine;
import java.io.File;

/** Android adapter factory shared by the camera and instrumentation tests. */
final class CabinetPipelines {
    static CabinetPipeline create(File directory, ModelChoice choice) throws Exception {
        long started = System.nanoTime();
        TextEngine engine = choice.ocr.usesMlKit() ? new MlKitTextEngine() : null;
        long sdkCreateNs = System.nanoTime() - started;
        // The pipeline owns and closes the adapter, including on constructor failure.
        CabinetPipeline pipeline = new CabinetPipeline(directory, choice, engine);
        if (engine != null) {
            pipeline.initializationTiming.add("model_load.platform_ocr", sdkCreateNs);
            pipeline.initializationTiming.add("model_load.total", sdkCreateNs);
        }
        return pipeline;
    }
}
