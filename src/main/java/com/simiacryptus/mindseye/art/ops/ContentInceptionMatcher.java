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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class ContentInceptionMatcher implements VisualModifier {

  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private boolean balanced = true;

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor content, Tensor... style) {
    network = network.copyPipeline();
    Tensor baseContent = network.eval(style).getDataAndFree().getAndFree(0);
    BandAvgReducerLayer bandAvgReducerLayer = new BandAvgReducerLayer();
    Tensor bandAvg = bandAvgReducerLayer.eval(baseContent).getDataAndFree().getAndFree(0);
    ImgBandBiasLayer offsetLayer = new ImgBandBiasLayer(bandAvg.scale(-1));
    Tensor bandPowers = PipelineNetwork.wrap(1,
        offsetLayer,
        new SquareActivationLayer(),
        bandAvgReducerLayer,
        new NthPowerActivationLayer().setPower(0.5)
    ).eval(baseContent).getDataAndFree().getAndFree(0);
    int[] contentDimensions = baseContent.getDimensions();
    Layer colorProjection = new ConvolutionLayer(1, 1, contentDimensions[2], 1)
        .setPaddingXY(0, 0)
        .setAndFree(bandPowers.unit())
        .explodeAndFree();
    Tensor spacialPattern = colorProjection.eval(baseContent).getDataAndFree().getAndFree(0);
    double mag = balanced ? spacialPattern.rms() : 1;
    network.wrap(colorProjection);
    network.wrap(new ConvolutionLayer(contentDimensions[0], contentDimensions[1], 1, 1)
        .setAndFree(spacialPattern.scaleInPlace(Math.pow(spacialPattern.rms(), -2))).explodeAndFree()).freeRef();
    network.wrap(PipelineNetwork.wrap(1,
        new BoundedActivationLayer().setMinValue(getMinValue()).setMaxValue(getMaxValue()),
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()
//        ,new NthPowerActivationLayer().setPower(0.5)
    ).setName(String.format("-RMS / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
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

  public int getMinValue() {
    return minValue;
  }

  public ContentInceptionMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public int getMaxValue() {
    return maxValue;
  }

  public ContentInceptionMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }
}
