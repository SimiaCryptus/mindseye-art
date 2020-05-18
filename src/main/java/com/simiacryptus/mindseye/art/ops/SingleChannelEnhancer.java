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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandSelectLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

/**
 * The type Single channel enhancer.
 */
public class SingleChannelEnhancer implements VisualModifier {

  private boolean averaging = true;
  private boolean balanced = true;
  private int minBand;
  private int maxBand;
  private double power = 2.0;

  /**
   * Instantiates a new Single channel enhancer.
   *
   * @param minBand the min band
   * @param maxBand the max band
   */
  public SingleChannelEnhancer(int minBand, int maxBand) {
    this.minBand = minBand;
    this.maxBand = maxBand;
  }

  /**
   * Gets power.
   *
   * @return the power
   */
  public double getPower() {
    return power;
  }

  /**
   * Sets power.
   *
   * @param power the power
   * @return the power
   */
  public SingleChannelEnhancer setPower(double power) {
    this.power = power;
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
  public SingleChannelEnhancer setAveraging(boolean averaging) {
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
  public SingleChannelEnhancer setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    double mag = 1;
    ImgBandSelectLayer selectLayer = new ImgBandSelectLayer(minBand, maxBand);
    if (balanced) {
      Tensor data0 = Result.getData0(selectLayer.eval(network.eval(visualModifierParameters.getStyle())));
      double rms = data0.rms();
      if (Double.isFinite(rms) && rms > 0.0) mag = rms;
      data0.freeRef();
    }
    Layer layer = PipelineNetwork.build(1,
        selectLayer,
        new NthPowerActivationLayer(getPower()),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new LinearActivationLayer(Math.pow(mag, -getPower())),
        new NthPowerActivationLayer(1.0 / getPower()),
        new LinearActivationLayer(-1)
    );
    layer.setName(RefString.format("-RMS / %.0E", mag));
    network.add(layer).freeRef();
    network.freeze();
    visualModifierParameters.freeRef();
    return network;
  }

}
