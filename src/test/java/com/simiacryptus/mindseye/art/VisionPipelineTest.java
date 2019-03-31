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

import com.simiacryptus.mindseye.art.models.InceptionVision;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class VisionPipelineTest {
  private static final Logger log = LoggerFactory.getLogger(VisionPipelineTest.class);

  public static class InceptionVisionTest extends VisionPipelineTest {
    @Test
    public void inoutDims() {
      testDims(InceptionVision.Layer1a, new int[]{320, 240, 3}, new int[]{160, 120, 64});
      testDims(InceptionVision.Layer2a, new int[]{160, 120, 64}, new int[]{80, 60, 192});
      testDims(InceptionVision.Layer3a, new int[]{80, 60, 192}, new int[]{40, 30, 256});
      testDims(InceptionVision.Layer3b, new int[]{40, 30, 256}, new int[]{40, 30, 480});
      testDims(InceptionVision.Layer4a, new int[]{40, 30, 480}, new int[]{20, 15, 508});
      testDims(InceptionVision.Layer4b, new int[]{20, 15, 508}, new int[]{20, 15, 512});
      testDims(InceptionVision.Layer4c, new int[]{20, 15, 512}, new int[]{20, 15, 512});
      testDims(InceptionVision.Layer4d, new int[]{20, 15, 512}, new int[]{20, 15, 528});
      testDims(InceptionVision.Layer4e, new int[]{20, 15, 528}, new int[]{20, 15, 832});
      testDims(InceptionVision.Layer5a, new int[]{20, 15, 832}, new int[]{10, 8, 832});
      testDims(InceptionVision.Layer5b, new int[]{10, 8, 832}, new int[]{10, 8, 1024});
    }

    @Test
    public void pipelineTest() {
      VisionPipeline<VisionPipelineLayer> pipeline = InceptionVision.getVisionPipeline();
      Assert.assertArrayEquals(testDims(pipeline, 320, 320, 3), new int[]{10, 10, 1024});
      Assert.assertArrayEquals(testDims(pipeline, 32, 32, 3), new int[]{1, 1, 1024});
      Assert.assertArrayEquals(testDims(pipeline, 400, 400, 3), new int[]{13, 13, 1024});
      Assert.assertArrayEquals(testDims(pipeline, 40, 40, 3), new int[]{2, 2, 1024});
    }

    @Test
    public void layerPins() {
      int sizeMultiplier = 4;
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer1a, new int[]{32 * sizeMultiplier, 24 * sizeMultiplier, 3});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer2a, new int[]{16 * sizeMultiplier, 12 * sizeMultiplier, 64});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer3a, new int[]{8 * sizeMultiplier, 6 * sizeMultiplier, 192});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer3b, new int[]{4 * sizeMultiplier, 3 * sizeMultiplier, 256});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer4a, new int[]{4 * sizeMultiplier, 3 * sizeMultiplier, 480});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer4b, new int[]{2 * sizeMultiplier, 2 * sizeMultiplier, 508});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer4c, new int[]{2 * sizeMultiplier, 2 * sizeMultiplier, 512});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer4d, new int[]{2 * sizeMultiplier, 2 * sizeMultiplier, 512});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer4e, new int[]{2 * sizeMultiplier, 2 * sizeMultiplier, 528});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer5a, new int[]{2 * sizeMultiplier, 2 * sizeMultiplier, 832});
      VisionPipelineUtil.testPinConnectivity(InceptionVision.Layer5b, new int[]{1 * sizeMultiplier, 1 * sizeMultiplier, 832});
    }

  }

  public static void testDims(VisionPipelineLayer inceptionVision, int[] inputDims, int[] expectedOutputDims) {
    Assert.assertArrayEquals(expectedOutputDims, inceptionVision.outputDims(inputDims));
    Assert.assertArrayEquals(expectedOutputDims, VisionPipelineUtil.evalDims(inputDims, inceptionVision.getLayer()));
  }

  public static int[] testDims(VisionPipelineLayer inceptionVision, int... inputDims) {
    int[] actuals = VisionPipelineUtil.evalDims(inputDims, inceptionVision.getLayer());
    Assert.assertArrayEquals(inceptionVision.outputDims(inputDims), actuals);
    return actuals;
  }

  public static int[] testDims(VisionPipeline<VisionPipelineLayer> pipeline, int... dims) {
    for (VisionPipelineLayer layer : pipeline.getLayers().keySet()) dims = testDims(layer, dims);
    return dims;
  }

}
