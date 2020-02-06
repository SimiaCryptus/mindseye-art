/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art;

import com.simiacryptus.mindseye.art.models.Inception5H;
import com.simiacryptus.mindseye.art.models.VGG16;
import com.simiacryptus.mindseye.art.models.VGG19;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.StandardLayerTests;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefAssert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;

import static com.simiacryptus.mindseye.art.models.Inception5H.*;
import static com.simiacryptus.mindseye.art.models.VGG16.*;
import static com.simiacryptus.mindseye.art.models.VGG19.*;

public abstract class VisionPipelineTest extends NotebookReportBase {
  private static final Logger log = LoggerFactory.getLogger(VisionPipelineTest.class);

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }

  @Nullable
  public abstract VisionPipeline<? extends VisionPipelineLayer> getVisionPipeline();

  public static void testDims(@Nonnull VisionPipelineLayer inceptionVision, int[] inputDims, @Nonnull int[] expectedOutputDims) {
    Layer layer = inceptionVision.getLayer();
    int[] dimensions = layer.evalDims(inputDims);
    layer.freeRef();
    int[] actuals = dimensions;
    RefAssert.assertArrayEquals(RefArrays.toString(actuals), expectedOutputDims, actuals);
  }

  @Nonnull
  public static int[] testDims(@Nonnull VisionPipelineLayer inceptionVision, int... inputDims) {
    Layer layer = inceptionVision.getLayer();
    int[] dimensions = layer.evalDims(inputDims);
    layer.freeRef();
    return dimensions;
  }

  public static int[] testDims(@Nonnull VisionPipeline<? extends VisionPipelineLayer> pipeline, int... dims) {
    for (VisionPipelineLayer layer : pipeline.getLayers().keySet())
      dims = testDims(layer, dims);
    return dims;
  }

  public int[] testDims(int... dims) {
    return testDims(getVisionPipeline(), dims);
  }

  @Test
  public void inoutDims() {
    run(log1 -> inoutDims(log1));
  }

  //  @Test
  //  public void layerPins() {
  //    run(this::layerPins);
  //  }

  @Test
  public void pipelineTest() {
    run(log1 -> pipelineTest(log1));
  }

  @Test
  public void graphs() {
    run(log1 -> graphs(log1));
  }

  @Test
  public void layers() {
    run(log1 -> layers(log1));
  }

  public abstract void inoutDims(NotebookOutput log);

  public abstract void pipelineTest(NotebookOutput log);

  public void graphs(@Nonnull NotebookOutput log) {
    getVisionPipeline().getLayers().keySet().forEach(e -> {
      log.h1(e.name());
      DAGNetwork layer = (DAGNetwork) e.getLayer();
      TestUtil.graph(log, layer);
      layer.freeRef();
    });
  }

  public void layers(@Nonnull NotebookOutput log) {
    final int[][] dims = {{226, 226, 3}};
    getVisionPipeline().getLayers().keySet().forEach(e -> {
      log.h1(e.name());
      DAGNetwork layer = (DAGNetwork) e.getLayer();
      log.subreport(sublog -> {
        new StandardLayerTests() {
          {
            testingBatchSize = 1;
            validateBatchExecution = false;
            validateDifferentials = false;
            testTraining = false;
          }

          @Override
          public Class<?> getTestClass() {
            return e.getClass();
          }

          @Nonnull
          @Override
          public int[][] getSmallDims(Random random) {
            return dims;
          }

          @Nonnull
          @Override
          public Layer getLayer(int[][] inputSize, Random random) {
            return layer.copy();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }

          @Nonnull
          @Override
          protected Layer lossLayer() {
            return new MeanSqLossLayer();
          }
        }.run(sublog);
        return null;
      }, log.getName() + "_" + e.name());
      Layer layer1 = e.getLayer();
      int[] dimensions = layer1.evalDims(dims[0]);
      layer1.freeRef();
      dims[0] = dimensions;
      layer.freeRef();
    });
  }

  public static class VGG16Test extends VisionPipelineTest {
    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return VGG16.class;
    }

    @Nullable
    @Override
    public VisionPipeline<? extends VisionPipelineLayer> getVisionPipeline() {
      return VGG16.getVisionPipeline();
    }

    @Nullable
    public static @SuppressWarnings("unused")
    VGG16Test[] addRefs(@Nullable VGG16Test[] array) {
      return RefUtil.addRefs(array);
    }

    public void inoutDims(@Nonnull NotebookOutput log) {
      log.run(() -> {
        testDims(VGG16_0b, new int[]{226, 226, 3}, new int[]{226, 226, 3});
        testDims(VGG16_1a, new int[]{226, 226, 3}, new int[]{226, 226, 64});
        testDims(VGG16_1b1, new int[]{226, 226, 64}, new int[]{113, 113, 128});
        testDims(VGG16_1c1, new int[]{113, 113, 128}, new int[]{57, 57, 256});
        testDims(VGG16_1d1, new int[]{57, 57, 256}, new int[]{29, 29, 512});
        testDims(VGG16_1e1, new int[]{29, 29, 512}, new int[]{15, 15, 512});
        testDims(VGG16_2, new int[]{15, 15, 512}, new int[]{8, 8, 512});
        testDims(VGG16_3a, new int[]{14, 14, 4096}, new int[]{14, 14, 1000});
        testDims(VGG16_3b, new int[]{14, 14, 1000}, new int[]{7, 7, 1000});
      });
    }

    @Override
    public void pipelineTest(@Nonnull NotebookOutput log) {
      log.run(() -> {
        int[] outputSize = testDims(226, 226, 3);
        RefAssert.assertArrayEquals(RefArrays.toString(outputSize), outputSize, new int[]{7, 7, 1000});
      });
    }

  }

  public static class VGG19Test extends VisionPipelineTest {
    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return VGG19.class;
    }

    @Nullable
    @Override
    public VisionPipeline<? extends VisionPipelineLayer> getVisionPipeline() {
      return VGG19.getVisionPipeline();
    }

    @Nullable
    public static @SuppressWarnings("unused")
    VGG19Test[] addRefs(@Nullable VGG19Test[] array) {
      return RefUtil.addRefs(array);
    }

    public void inoutDims(@Nonnull NotebookOutput log) {
      log.run(() -> {
        testDims(VGG19_0b, new int[]{226, 226, 3}, new int[]{226, 226, 3});
        testDims(VGG19_1a, new int[]{226, 226, 3}, new int[]{226, 226, 64});
        testDims(VGG19_1b1, new int[]{226, 226, 64}, new int[]{113, 113, 128});
        testDims(VGG19_1c1, new int[]{113, 113, 128}, new int[]{57, 57, 256});
        testDims(VGG19_1d1, new int[]{57, 57, 256}, new int[]{29, 29, 512});
        testDims(VGG19_1e1, new int[]{29, 29, 512}, new int[]{15, 15, 512});
        testDims(VGG19_2, new int[]{8, 8, 512}, new int[]{14, 14, 4096});
        //        testDims(VGG19_3a, new int[]{14, 14, 4096}, new int[]{14, 14, 1000});
        //        testDims(VGG19_3b, new int[]{14, 14, 1000}, new int[]{7, 7, 1000});
      });
    }

    @Override
    public void pipelineTest(@Nonnull NotebookOutput log) {
      log.run(() -> {
        int[] outputSize = testDims(226, 226, 3);
        RefAssert.assertArrayEquals(RefArrays.toString(outputSize), outputSize, new int[]{7, 7, 1000});
      });
    }

  }

  public static class Inception5HTest extends VisionPipelineTest {
    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return Inception5H.class;
    }

    @Nullable
    @Override
    public VisionPipeline<? extends VisionPipelineLayer> getVisionPipeline() {
      return Inception5H.getVisionPipeline();
    }

    public void inoutDims(@Nonnull NotebookOutput log) {
      log.run(() -> {
        testDims(Inc5H_1a, new int[]{320, 240, 3}, new int[]{160, 120, 64});
        testDims(Inc5H_2a, new int[]{160, 120, 64}, new int[]{80, 60, 192});
        testDims(Inc5H_3a, new int[]{80, 60, 192}, new int[]{40, 30, 256});
        testDims(Inc5H_3b, new int[]{40, 30, 256}, new int[]{40, 30, 480});
        testDims(Inc5H_4a, new int[]{40, 30, 480}, new int[]{20, 15, 508});
        testDims(Inc5H_4b, new int[]{20, 15, 508}, new int[]{20, 15, 512});
        testDims(Inc5H_4c, new int[]{20, 15, 512}, new int[]{20, 15, 512});
        testDims(Inc5H_4d, new int[]{20, 15, 512}, new int[]{20, 15, 528});
        testDims(Inc5H_4e, new int[]{20, 15, 528}, new int[]{20, 15, 832});
        testDims(Inc5H_5a, new int[]{20, 15, 832}, new int[]{10, 8, 832});
        testDims(Inc5H_5b, new int[]{10, 8, 832}, new int[]{10, 8, 1024});
      });
    }

    public void pipelineTest(@Nonnull NotebookOutput log) {
      log.run(() -> {
        RefAssert.assertArrayEquals(testDims(320, 320, 3), new int[]{10, 10, 1024});
      });
      log.run(() -> {
        RefAssert.assertArrayEquals(testDims(32, 32, 3), new int[]{1, 1, 1024});
      });
      log.run(() -> {
        RefAssert.assertArrayEquals(testDims(400, 400, 3), new int[]{13, 13, 1024});
      });
      log.run(() -> {
        RefAssert.assertArrayEquals(testDims(40, 40, 3), new int[]{2, 2, 1024});
      });
    }

  }

}
