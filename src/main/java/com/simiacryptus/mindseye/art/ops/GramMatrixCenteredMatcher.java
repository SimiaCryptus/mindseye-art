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

import com.simiacryptus.mindseye.art.ArtSettings;
import com.simiacryptus.mindseye.art.TiledTrainable;
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
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
import java.util.function.Function;

/**
 * The type Gram matrix centered matcher.
 */
public class GramMatrixCenteredMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixCenteredMatcher.class);
  private final Precision precision = Precision.Float;
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = ArtSettings.INSTANCE().defaultTileSize;

  /**
   * Gets tile size.
   *
   * @return the tile size
   */
  public int getTileSize() {
    return tileSize;
  }

  /**
   * Sets tile size.
   *
   * @param tileSize the tile size
   * @return the tile size
   */
  @Nonnull
  public GramMatrixCenteredMatcher setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  /**
   * Is averaging boolean.
   *
   * @return the boolean
   */
  public boolean isAveraging() {
    return averaging;
  }

  /**
   * Sets averaging.
   *
   * @param averaging the averaging
   * @return the averaging
   */
  @Nonnull
  public GramMatrixCenteredMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return balanced;
  }

  /**
   * Sets balanced.
   *
   * @param balanced the balanced
   * @return the balanced
   */
  @Nonnull
  public GramMatrixCenteredMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  /**
   * Loss layer.
   *
   * @param result    the result
   * @param mag       the mag
   * @param averaging the averaging
   * @return the layer
   */
  @Nonnull
  public static Layer loss(@Nonnull Tensor result, double mag, boolean averaging) {
    result.scaleInPlace(-1);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(Math.pow(mag, -2));
    Layer layer1 = PipelineNetwork.build(1,
        new ImgBandBiasLayer(result),
        new SquareActivationLayer(),
        averaging ? new AvgReducerLayer() : new SumReducerLayer(),
        linearActivationLayer);
    layer1.setName(RefString.format("RMS[x-C] / %.0E", mag));
    return layer1;
  }

  /**
   * Eval tensor.
   *
   * @param pixels   the pixels
   * @param network  the network
   * @param tileSize the tile size
   * @param image    the image
   * @return the tensor
   */
  @Nonnull
  public static Tensor eval(int pixels, @Nonnull PipelineNetwork network, int tileSize, @Nonnull Tensor... image) {
    Tensor tensor1 = RefUtil.get(RefArrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      return RefArrays.stream(TiledTrainable.selectors(0, imageDimensions[0], imageDimensions[1], tileSize, false))
          .map(RefUtil.wrapInterface((Function<Layer, Tensor>) selector -> {
            //log.info(selector.toString());
            Tensor tile = Result.getData0(selector.eval(img.addRef()));
            selector.freeRef();
            int[] tileDimensions = tile.getDimensions();
            Tensor tensor = Result.getData0(network.eval(tile));
            tensor.scaleInPlace(tileDimensions[0] * tileDimensions[1]);
            return tensor;
          }, img));
    }).reduce((a, b) -> {
      a.addInPlace(b);
      return a;
    }));
    network.freeRef();
    tensor1.scaleInPlace(1.0 / pixels);
    //log.info(selector.toString());
    Tensor map = tensor1.map(x -> {
      if (Double.isFinite(x)) {
        return x;
      } else {
        return 0;
      }
    });
    tensor1.freeRef();
    return map;
  }

  /**
   * Gets append uuid.
   *
   * @param network    the network
   * @param layerClass the layer class
   * @return the append uuid
   */
  @Nonnull
  public static UUID getAppendUUID(@Nonnull PipelineNetwork network, @Nonnull Class<GramianLayer> layerClass) {
    DAGNode head = network.getHead();
    network.freeRef();
    Layer layer = head.getLayer();
    head.freeRef();
    if (null == layer)
      return UUID.randomUUID();
    UUID uuid = UUID.nameUUIDFromBytes((layer.getId().toString() + layerClass.getName()).getBytes());
    layer.freeRef();
    return uuid;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    final PipelineNetwork pipelineNetwork = buildWithModel(visualModifierParameters.getNetwork(), null,
        visualModifierParameters.getStyle());
    visualModifierParameters.freeRef();
    return pipelineNetwork;
  }

  /**
   * Build with model pipeline network.
   *
   * @param network the network
   * @param cov     the cov
   * @param images  the images
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork buildWithModel(PipelineNetwork network, @Nullable Tensor cov, @Nonnull Tensor... images) {
    PipelineNetwork copyPipeline = network.copyPipeline();
    RefUtil.freeRef(network);
    network = copyPipeline;
    MultiPrecision.setPrecision(network.addRef(), precision);
    assert network != null;
    GramianLayer gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network.addRef(), GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(precision);
    network.add(gramianLayerMultiPrecision).freeRef();
    int pixels = RefArrays.stream(RefUtil.addRef(images)).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      x.freeRef();
      return dimensions[0] * dimensions[1];
    }).sum();
    if (null == cov) {
      RefUtil.freeRef(cov);
      cov = eval(pixels == 0 ? 1 : pixels, network.addRef(), getTileSize(), images);
    } else {
      RefUtil.freeRef(images);
    }
    double mag = balanced ? cov.rms() : 1;
    network.add(loss(cov, mag, isAveraging())).freeRef();
    network.freeze();
    return network;
  }
}
