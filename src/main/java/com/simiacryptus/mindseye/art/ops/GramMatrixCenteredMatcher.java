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

package com.simiacryptus.mindseye.art.ops;

import com.simiacryptus.mindseye.art.TiledTrainable;
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

public class GramMatrixCenteredMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixCenteredMatcher.class);
  private final Precision precision = Precision.Float;
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = 600;

  public int getTileSize() {
    return tileSize;
  }

  @Nonnull
  public GramMatrixCenteredMatcher setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public GramMatrixCenteredMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public GramMatrixCenteredMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  public static Layer loss(@Nonnull Tensor result, double mag, boolean averaging) {
    result.scaleInPlace(-1);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(Math.pow(mag, -2));
    final Layer[] layers = new Layer[]{new ImgBandBiasLayer(result.addRef()), new SquareActivationLayer(),
        averaging ? new AvgReducerLayer() : new SumReducerLayer(),
        linearActivationLayer.addRef()};
    Layer layer1 = PipelineNetwork.build(1, layers);
    layer1.setName(RefString.format("RMS[x-C] / %.0E", mag));
    Layer layer = layer1.addRef();
    result.freeRef();
    return layer;
  }

  @Nonnull
  public static Tensor eval(int pixels, @Nonnull PipelineNetwork network, int tileSize, @Nonnull Tensor... image) {
    Tensor tensor1 = RefUtil.get(RefArrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      return RefArrays.stream(TiledTrainable.selectors(0, imageDimensions[0], imageDimensions[1], tileSize, false))
          .map(selector -> {
            //log.info(selector.toString());
            Tensor tile = selector.eval(img).getData().get(0);
            selector.freeRef();
            int[] tileDimensions = tile.getDimensions();
            Tensor tensor = network.eval(tile).getData().get(0);
            tensor.scaleInPlace(tileDimensions[0] * tileDimensions[1]);
            Tensor component = tensor.addRef();
            tile.freeRef();
            return component;
          });
    }).reduce((a, b) -> {
      a.addInPlace(b);
      b.freeRef();
      return a;
    }));
    tensor1.scaleInPlace(1.0 / pixels);
    //log.info(selector.toString());
    return tensor1.addRef().map(x -> {
      if (Double.isFinite(x)) {
        return x;
      } else {
        return 0;
      }
    });
  }

  @Nonnull
  public static UUID getAppendUUID(@Nonnull PipelineNetwork network, @Nonnull Class<GramianLayer> layerClass) {
    DAGNode head = network.getHead();
    Layer layer = head.getLayer();
    if (null == layer)
      return UUID.randomUUID();
    return UUID.nameUUIDFromBytes((layer.getId().toString() + layerClass.getName()).getBytes());
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    final PipelineNetwork pipelineNetwork = buildWithModel(visualModifierParameters.network, null,
        visualModifierParameters.style);
    visualModifierParameters.freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  public PipelineNetwork buildWithModel(PipelineNetwork network, @Nullable Tensor cov, @Nonnull Tensor... image) {
    network = network.copyPipeline();
    MultiPrecision.setPrecision(network, precision);
    assert network != null;
    MultiPrecision gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network, GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(precision);
    network.add((GramianLayer) RefUtil.addRef(gramianLayerMultiPrecision)).freeRef();
    int pixels = RefArrays.stream(image).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum();
    if (null == cov)
      cov = eval(pixels == 0 ? 1 : pixels, network, getTileSize(), image);
    double mag = balanced ? cov.rms() : 1;
    network.add(loss(cov, mag, isAveraging())).freeRef();
    network.freeze();
    return network.addRef();
  }
}
