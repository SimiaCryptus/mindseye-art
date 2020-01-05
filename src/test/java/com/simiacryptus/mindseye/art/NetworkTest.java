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
import com.simiacryptus.mindseye.layers.java.LayerTestBase;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Random;

public @RefAware
class NetworkTest extends LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(NetworkTest.class);
  private static final BufferedImage styleImage = ImageArtUtil.load(new NullNotebookOutput(),
      "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg", 1200);
  private static final BufferedImage contentImage = ImageArtUtil.load(new NullNotebookOutput(),
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

  public static @SuppressWarnings("unused")
  NetworkTest[] addRefs(NetworkTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NetworkTest::addRef)
        .toArray((x) -> new NetworkTest[x]);
  }

  public static @SuppressWarnings("unused")
  NetworkTest[][] addRefs(NetworkTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(NetworkTest::addRefs)
        .toArray((x) -> new NetworkTest[x][]);
  }

  private static DAGNetwork build() {
    Tensor styleTensor = Tensor.fromRGB(styleImage);
    DAGNetwork dagNetwork = MultiPrecision.setPrecision(
        SumInputsLayer.combine(new GramMatrixMatcher().build(Inception5H.Inc5H_2a, null, null, styleTensor),
            new GramMatrixMatcher().build(Inception5H.Inc5H_3a, null, null, styleTensor),
            new ContentMatcher().build(VisionPipelineLayer.NOOP, null, null, Tensor.fromRGB(contentImage))
                .andThenWrap(new LinearActivationLayer().setScale(1e-1).freeze())),
        Precision.Float);
    styleTensor.freeRef();
    return dagNetwork;
  }

  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{contentImage.getWidth(), contentImage.getHeight(), 3}};
  }

  @Override
  public Layer getLayer(int[][] inputSize, Random random) {
    return layer.copy();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  NetworkTest addRef() {
    return (NetworkTest) super.addRef();
  }
}
