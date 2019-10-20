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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class ContentConvolutionMatcher implements VisualModifier {

  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private boolean balanced = true;
  private int patternSize = 32 * 32 * 3;
  private PoolingLayer.PoolingMode poolingMode = PoolingLayer.PoolingMode.Max;

  @Override
  public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    network = network.copyPipeline();
    Tensor baseContent = network.eval(visualModifierParameters.style).getDataAndFree().getAndFree(0);
    visualModifierParameters.freeRef();
    double mag = balanced ? baseContent.rms() : 1;
    int[] baseContentDimensions = baseContent.getDimensions();
    int patternSize = (int) Math.ceil(Math.sqrt(getPatternSize() / baseContentDimensions[2]));
    PoolingLayer poolingLayer = new PoolingLayer().setMode(getPoolingMode())
        .setStrideXY((int) Math.max(1, Math.floor((double) baseContentDimensions[0] / patternSize)), (int) Math.max(1, Math.floor((double) baseContentDimensions[1] / patternSize)))
        .setWindowXY((int) Math.max(1, Math.floor((double) baseContentDimensions[0] / patternSize)), (int) Math.max(1, Math.floor((double) baseContentDimensions[1] / patternSize)));
    Tensor pooledContent = poolingLayer.eval(baseContent).getDataAndFree().getAndFree(0);
    baseContent.freeRef();
    network.wrap(poolingLayer).freeRef();
    int[] pooledContentDimensions = pooledContent.getDimensions();
    network.wrap(new ConvolutionLayer(pooledContentDimensions[0], pooledContentDimensions[1], pooledContentDimensions[2], 1)
        .setPaddingXY(0, 0)
        .setAndFree(pooledContent
            .scaleInPlace(Math.pow(pooledContent.rms(), -2))
            .permuteDimensionsAndFree(Integer.MAX_VALUE, -1, 2)
        ).explodeAndFree()).freeRef();
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

  public ContentConvolutionMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ContentConvolutionMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  public int getPatternSize() {
    return patternSize;
  }

  public ContentConvolutionMatcher setPatternSize(int patternSize) {
    this.patternSize = patternSize;
    return this;
  }

  public PoolingLayer.PoolingMode getPoolingMode() {
    return poolingMode;
  }

  public ContentConvolutionMatcher setPoolingMode(PoolingLayer.PoolingMode poolingMode) {
    this.poolingMode = poolingMode;
    return this;
  }

  public int getMinValue() {
    return minValue;
  }

  public ContentConvolutionMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public int getMaxValue() {
    return maxValue;
  }

  public ContentConvolutionMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }
}
