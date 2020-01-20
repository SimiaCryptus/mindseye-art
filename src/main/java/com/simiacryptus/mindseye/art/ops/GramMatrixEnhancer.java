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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.UUID;

public class GramMatrixEnhancer implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixEnhancer.class);
  private final Precision precision = Precision.Float;
  private double min = -1;
  private double max = 1;
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = 600;
  private int padding = 8;

  public double getMax() {
    return max;
  }

  @Nonnull
  public GramMatrixEnhancer setMax(double max) {
    this.max = max;
    return this;
  }

  public double getMin() {
    return min;
  }

  @Nonnull
  public GramMatrixEnhancer setMin(double min) {
    this.min = min;
    return this;
  }

  public int getTileSize() {
    return tileSize;
  }

  @Nonnull
  public GramMatrixEnhancer setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public GramMatrixEnhancer setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public GramMatrixEnhancer setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
    PipelineNetwork rmsNetwork = new PipelineNetwork(1);
    rmsNetwork.setName(RefString.format("-RMS[x*C] / %.0E", mag));
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = -Math.pow(mag, -2);
    linearActivationLayer.setScale(scale);
    final Layer nextHead = linearActivationLayer.addRef();
    final Layer nextHead1 = averaging ? new AvgReducerLayer() : new SumReducerLayer();
    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(getMin());
    BoundedActivationLayer boundedActivationLayer = boundedActivationLayer1.addRef();
    boundedActivationLayer.setMaxValue(getMax());
    rmsNetwork
        .add(nextHead1,
            rmsNetwork.add(boundedActivationLayer.addRef(),
                rmsNetwork.add(nextHead,
                    rmsNetwork.add(new ProductLayer(), rmsNetwork.getInput(0), rmsNetwork.constValueWrap(result)))))
        .freeRef();
    return rmsNetwork;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    assert network != null;
    network = network.copyPipeline();
    MultiPrecision.setPrecision(network, precision);
    assert network != null;
    final UUID uuid = GramMatrixMatcher.getAppendUUID(network, GramianLayer.class);
    int pixels = RefArrays.stream(visualModifierParameters.style).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum();

    final PipelineNetwork copy = network.copyPipeline();
    assert copy != null;
    MultiPrecision gramianLayerMultiPrecision1 = new GramianLayer(uuid);
    gramianLayerMultiPrecision1.setPrecision(precision);
    copy.add((GramianLayer) RefUtil.addRef(gramianLayerMultiPrecision1)).freeRef();
    Tensor result = GramMatrixMatcher.eval(pixels, copy, getTileSize(), padding, visualModifierParameters.style);
    copy.freeRef();

    final Tensor boolMask = MomentMatcher
        .toMask(MomentMatcher.transform(network, visualModifierParameters.mask, Precision.Float));
    network.add(new ProductLayer(), network.getHead(), network.constValue(boolMask)).freeRef();
    visualModifierParameters.freeRef();
    MultiPrecision gramianLayerMultiPrecision = new GramianLayer(uuid);
    gramianLayerMultiPrecision.setPrecision(precision);
    network.add((GramianLayer) RefUtil.addRef(gramianLayerMultiPrecision)).freeRef();

    assert result != null;
    double mag = balanced ? result.rms() : 1;
    network.add(loss(result, mag, isAveraging())).freeRef();
    network.freeze();
    return network.addRef();
  }

  @Nonnull
  public GramMatrixEnhancer setMinMax(double minValue, double maxValue) {
    this.min = minValue;
    this.max = maxValue;
    return this;
  }

  public static class StaticGramMatrixEnhancer extends GramMatrixEnhancer {
    @Nonnull
    public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
      PipelineNetwork rmsNetwork = new PipelineNetwork(1);
      rmsNetwork.setName(RefString.format("-RMS[x*C] / %.0E", mag));
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      final double scale = -Math.pow(mag, -2);
      linearActivationLayer.setScale(scale);
      final Layer nextHead = linearActivationLayer.addRef();
      final Layer nextHead1 = averaging ? new AvgReducerLayer() : new SumReducerLayer();
      BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
      boundedActivationLayer1.setMinValue(getMin());
      BoundedActivationLayer boundedActivationLayer = boundedActivationLayer1.addRef();
      boundedActivationLayer.setMaxValue(getMax());
      rmsNetwork.add(nextHead1, rmsNetwork.add(boundedActivationLayer.addRef(),
          rmsNetwork.add(nextHead, rmsNetwork.add(new AbsActivationLayer())))).freeRef();
      return rmsNetwork;
    }
  }
}
