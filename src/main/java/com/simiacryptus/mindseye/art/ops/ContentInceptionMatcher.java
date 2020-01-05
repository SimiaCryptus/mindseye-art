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
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;

public @RefAware
class ContentInceptionMatcher implements VisualModifier {

  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private boolean balanced = true;

  public int getMaxValue() {
    return maxValue;
  }

  public ContentInceptionMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  public int getMinValue() {
    return minValue;
  }

  public ContentInceptionMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  public ContentInceptionMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ContentInceptionMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Override
  public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    network = network.copyPipeline();
    Tensor baseContent = network.eval(visualModifierParameters.style).getData().get(0);
    visualModifierParameters.freeRef();
    BandAvgReducerLayer bandAvgReducerLayer = new BandAvgReducerLayer();
    Tensor bandAvg = bandAvgReducerLayer.eval(baseContent).getData().get(0);
    ImgBandBiasLayer offsetLayer = new ImgBandBiasLayer(bandAvg.scale(-1));
    Tensor bandPowers = PipelineNetwork.build(1, offsetLayer, new SquareActivationLayer(), bandAvgReducerLayer,
        new NthPowerActivationLayer().setPower(0.5)).eval(baseContent).getData().get(0);
    int[] contentDimensions = baseContent.getDimensions();
    Layer colorProjection = new ConvolutionLayer(1, 1, contentDimensions[2], 1).setPaddingXY(0, 0)
        .set(bandPowers.unit()).explode();
    Tensor spacialPattern = colorProjection.eval(baseContent).getData().get(0);
    double mag = balanced ? spacialPattern.rms() : 1;
    network.add(colorProjection);
    network.add(new ConvolutionLayer(contentDimensions[0], contentDimensions[1], 1, 1)
        .set(spacialPattern.scaleInPlace(Math.pow(spacialPattern.rms(), -2))).explode()).freeRef();
    final Layer[] layers = new Layer[]{
        new BoundedActivationLayer().setMinValue(getMinValue()).setMaxValue(getMaxValue()), new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()};
    network.add(PipelineNetwork.build(1, layers).setName(String.format("-RMS / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
  }
}
