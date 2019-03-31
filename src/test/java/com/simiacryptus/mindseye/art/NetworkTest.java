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

import com.simiacryptus.mindseye.art.constraints.GramMatrixMatcher;
import com.simiacryptus.mindseye.art.constraints.RMSContentMatcher;
import com.simiacryptus.mindseye.art.models.InceptionVision;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.Random;

public class NetworkTest extends LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(NetworkTest.class);

  public NetworkTest() {
    validateDifferentials = false;
    validateBatchExecution = false;
    testTraining = false;
    testEquivalency = false;
    testingBatchSize = 1;
  }

  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{ { contentImage.getWidth(),contentImage.getHeight(),3 } };
  }

  private static final BufferedImage styleImage = VisionPipelineUtil.load("https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200);
  private static final BufferedImage contentImage = VisionPipelineUtil.load("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg", 500);

  private static final DAGNetwork layer = build();

  private static DAGNetwork build() {
    Tensor styleTensor = Tensor.fromRGB(styleImage);
    DAGNetwork dagNetwork = MultiPrecision.setPrecision(SumInputsLayer.combine(
        new GramMatrixMatcher().build(InceptionVision.Layer2a.getNetwork(),
            styleTensor
        ),
        new GramMatrixMatcher().build(InceptionVision.Layer3a.getNetwork(),
            styleTensor
        ),
        new RMSContentMatcher().build(new PipelineNetwork(),
            Tensor.fromRGB(contentImage)
        ).andThenWrap(new LinearActivationLayer().setScale(1e-1).freeze())
    ), Precision.Float);
    styleTensor.freeRef();
    return dagNetwork;
  }

  @Override
  public Layer getLayer(int[][] inputSize, Random random) {
    return layer.copy();
  }
}
