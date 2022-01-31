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
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.java.AbsActivationLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.UUID;

/**
 * The type Gram matrix enhancer.
 */
public class GramMatrixEnhancer implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixEnhancer.class);
  private final Precision precision = Precision.Float;
  private double min = -1;
  private double max = 1;
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = ArtSettings.INSTANCE().defaultTileSize;
  private int padding = 8;

  /**
   * Gets max.
   *
   * @return the max
   */
  public double getMax() {
    return max;
  }

  /**
   * Sets max.
   *
   * @param max the max
   * @return the max
   */
  @Nonnull
  public GramMatrixEnhancer setMax(double max) {
    this.max = max;
    return this;
  }

  /**
   * Gets min.
   *
   * @return the min
   */
  public double getMin() {
    return min;
  }

  /**
   * Sets min.
   *
   * @param min the min
   * @return the min
   */
  @Nonnull
  public GramMatrixEnhancer setMin(double min) {
    this.min = min;
    return this;
  }

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
  public GramMatrixEnhancer setTileSize(int tileSize) {
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
  public GramMatrixEnhancer setAveraging(boolean averaging) {
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
  public GramMatrixEnhancer setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  /**
   * Loss pipeline network.
   *
   * @param result    the result
   * @param mag       the mag
   * @param averaging the averaging
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
    PipelineNetwork rmsNetwork = new PipelineNetwork(1);
    rmsNetwork.setName(RefString.format("-RMS[x*C] / %.0E", mag));
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = mag == 0 ? 1 : -Math.pow(mag, -2);
    linearActivationLayer.setScale(scale);
    final Layer nextHead1 = averaging ? new AvgReducerLayer() : new SumReducerLayer();
    BoundedActivationLayer boundedActivationLayer = new BoundedActivationLayer();
    boundedActivationLayer.setMinValue(getMin());
    boundedActivationLayer.setMaxValue(getMax());
    rmsNetwork
        .add(nextHead1,
            rmsNetwork.add(boundedActivationLayer,
                rmsNetwork.add(linearActivationLayer,
                    rmsNetwork.add(new ProductLayer(), rmsNetwork.getInput(0), rmsNetwork.constValueWrap(result)))))
        .freeRef();
    return rmsNetwork;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    MultiPrecision.setPrecision(network.addRef(), precision);
    final UUID uuid = GramMatrixMatcher.getAppendUUID(network.addRef(), GramianLayer.class);
    int pixels = RefArrays.stream(visualModifierParameters.getStyle()).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      x.freeRef();
      return dimensions[0] * dimensions[1];
    }).sum();

    final PipelineNetwork copy = network.copyPipeline();
    assert copy != null;
    GramianLayer gramianLayerMultiPrecision1 = new GramianLayer(uuid);
    gramianLayerMultiPrecision1.setPrecision(precision);
    copy.add(gramianLayerMultiPrecision1).freeRef();
    Tensor result = GramMatrixMatcher.eval(pixels, copy, getTileSize(), padding, visualModifierParameters.getStyle());

    Tensor mask = visualModifierParameters.getMask();
    if (mask != null) {
      network.add(new ProductLayer(),
          network.getHead(),
          network.constValue(
              MomentMatcher.toMask(MomentMatcher.transform(network.addRef(), mask, Precision.Float))
          )
      ).freeRef();
    }

    GramianLayer gramianLayerMultiPrecision = new GramianLayer(uuid);
    gramianLayerMultiPrecision.setPrecision(precision);
    network.add(gramianLayerMultiPrecision).freeRef();

    assert result != null;
    double mag = balanced ? result.rms() : 1;
    log.info(RefString.format("Adjust for %s by %s: %s", network.getName(), this.getClass().getSimpleName(), mag));
    network.add(loss(result, mag, isAveraging())).freeRef();
    network.freeze();

    {
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      linearActivationLayer.setScale(visualModifierParameters.scale);
      linearActivationLayer.freeze();
      network.add(linearActivationLayer).freeRef();
    }

    visualModifierParameters.freeRef();
    return network;
  }

  /**
   * Sets min max.
   *
   * @param minValue the min value
   * @param maxValue the max value
   * @return the min max
   */
  @Nonnull
  public GramMatrixEnhancer setMinMax(double minValue, double maxValue) {
    this.min = minValue;
    this.max = maxValue;
    return this;
  }

  /**
   * The type Static gram matrix enhancer.
   */
  public static class StaticGramMatrixEnhancer extends GramMatrixEnhancer {
    @Nonnull
    public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
      result.freeRef();
      PipelineNetwork rmsNetwork = new PipelineNetwork(1);
      rmsNetwork.setName(RefString.format("-RMS[x*C] / %.0E", mag));
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      final double scale = -Math.pow(mag, -2);
      linearActivationLayer.setScale(scale);
      final Layer nextHead1 = averaging ? new AvgReducerLayer() : new SumReducerLayer();
      BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
      boundedActivationLayer1.setMinValue(getMin());
      boundedActivationLayer1.setMaxValue(getMax());
      rmsNetwork.add(nextHead1, rmsNetwork.add(boundedActivationLayer1,
          rmsNetwork.add(linearActivationLayer, rmsNetwork.add(new AbsActivationLayer())))).freeRef();
      return rmsNetwork;
    }
  }
}
