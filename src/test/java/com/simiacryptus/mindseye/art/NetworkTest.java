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
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.notebook.NullNotebookOutput;
import org.junit.jupiter.api.Disabled;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;

/**
 * The type Network test.
 */
public class NetworkTest extends LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(NetworkTest.class);
  private static final BufferedImage styleImage = ImageArtUtil.loadImage(new NullNotebookOutput(),
      "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200);
  private static final BufferedImage contentImage = ImageArtUtil.loadImage(new NullNotebookOutput(),
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg",
      500);
  private static final DAGNetwork layer = build();

  /**
   * Instantiates a new Network test.
   */
  public NetworkTest() {
    testingBatchSize = 1;
  }

  @Nonnull
  @Override
  public Layer getLayer() {
    return layer.copy();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{contentImage.getWidth(), contentImage.getHeight(), 3}};
  }

  private static DAGNetwork build() {
    Tensor styleTensor = Tensor.fromRGB(styleImage);
    DAGNetwork dagNetwork = SumInputsLayer.combine(
        new GramMatrixMatcher().build(Inception5H.Inc5H_2a, null, null, styleTensor.addRef()),
        new GramMatrixMatcher().build(Inception5H.Inc5H_3a, null, null, styleTensor),
        new ContentMatcher().scale(1e-1).build(VisionPipelineLayer.NOOP, null, null, Tensor.fromRGB(contentImage)));
    MultiPrecision.setPrecision(dagNetwork.addRef(), Precision.Float);
    return dagNetwork;
  }

  @Override
  @Disabled
  public void derivativeTest() {
    super.derivativeTest();
  }

  @Override
  @Disabled
  public void equivalencyTest() {
    super.equivalencyTest();
  }

  @Override
  @Disabled
  public void trainingTest() {
    super.trainingTest();
  }

  @Override
  @Disabled
  public void batchingTest() {
    super.batchingTest();
  }

}
