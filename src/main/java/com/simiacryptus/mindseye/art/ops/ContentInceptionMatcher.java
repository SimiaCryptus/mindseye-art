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
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

public class ContentInceptionMatcher implements VisualModifier {

  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private boolean balanced = true;

  public int getMaxValue() {
    return maxValue;
  }

  @Nonnull
  public ContentInceptionMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  public int getMinValue() {
    return minValue;
  }

  @Nonnull
  public ContentInceptionMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public ContentInceptionMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public ContentInceptionMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    Tensor baseContent = Result.getData0(network.eval(visualModifierParameters.getStyle()));
    visualModifierParameters.freeRef();
    BandAvgReducerLayer bandAvgReducerLayer = new BandAvgReducerLayer();
    Tensor bandAvg = Result.getData0(bandAvgReducerLayer.eval(baseContent.addRef()));
    ImgBandBiasLayer offsetLayer = new ImgBandBiasLayer(bandAvg.scale(-1));
    bandAvg.freeRef();
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(0.5);
    PipelineNetwork build = PipelineNetwork.build(1, offsetLayer, new SquareActivationLayer(), bandAvgReducerLayer,
        nthPowerActivationLayer);
    Tensor bandPowers = Result.getData0(build.eval(baseContent.addRef()));
    build.freeRef();
    int[] contentDimensions = baseContent.getDimensions();
    ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(1, 1, contentDimensions[2], 1);
    convolutionLayer2.setPaddingXY(0, 0);
    convolutionLayer2.set(bandPowers.unit());
    bandPowers.freeRef();
    Layer colorProjection = convolutionLayer2.explode();
    convolutionLayer2.freeRef();
    Tensor spacialPattern = Result.getData0(colorProjection.eval(baseContent));
    double mag = balanced ? spacialPattern.rms() : 1;
    network.add(colorProjection).freeRef();
    spacialPattern.scaleInPlace(Math.pow(spacialPattern.rms(), -2));
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(contentDimensions[0], contentDimensions[1], 1, 1);
    convolutionLayer.set(spacialPattern);
    network.add(convolutionLayer.explode()).freeRef();
    convolutionLayer.freeRef();
    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(getMinValue());
    boundedActivationLayer1.setMaxValue(getMaxValue());
    final Layer[] layers = new Layer[]{
        boundedActivationLayer1, new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()};
    Layer layer = PipelineNetwork.build(1, layers);
    layer.setName(RefString.format("-RMS / %.0E", mag));
    network.add(layer).freeRef();
    network.freeze();
    return network;
  }
}
