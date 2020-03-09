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
import com.simiacryptus.mindseye.art.ops.ContentMatcher;
import com.simiacryptus.mindseye.art.ops.GramMatrixMatcher;
import com.simiacryptus.mindseye.art.util.ImageArtUtil;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NullNotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Random;

public class NetworkTest extends LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(NetworkTest.class);
  private static final BufferedImage styleImage = ImageArtUtil.loadImage(new NullNotebookOutput(),
      "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200);
  private static final BufferedImage contentImage = ImageArtUtil.loadImage(new NullNotebookOutput(),
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
      500);
  private static final DAGNetwork layer = build();

  public NetworkTest() {
    validateDifferentials = false;
    validateBatchExecution = false;
    testTraining = false;
    testEquivalency = false;
    testingBatchSize = 1;
  }


  private static DAGNetwork build() {
    Tensor styleTensor = Tensor.fromRGB(styleImage);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(1e-1);
    linearActivationLayer.freeze();
    PipelineNetwork build = new ContentMatcher().build(VisionPipelineLayer.NOOP, null, null, Tensor.fromRGB(contentImage));
    DAGNetwork dagNetwork = SumInputsLayer.combine(
        new GramMatrixMatcher().build(Inception5H.Inc5H_2a, null, null, styleTensor.addRef()),
        new GramMatrixMatcher().build(Inception5H.Inc5H_3a, null, null, styleTensor.addRef()),
        build.andThen(linearActivationLayer));
    styleTensor.freeRef();
    build.freeRef();
    MultiPrecision.setPrecision(dagNetwork.addRef(), Precision.Float);
    return dagNetwork;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{contentImage.getWidth(), contentImage.getHeight(), 3}};
  }

  @Nonnull
  @Override
  public Layer getLayer(int[][] inputSize, Random random) {
    return layer.copy();
  }

}
