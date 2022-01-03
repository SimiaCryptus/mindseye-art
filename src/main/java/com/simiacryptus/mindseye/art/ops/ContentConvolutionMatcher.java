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
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;

/**
 * The type Content convolution matcher.
 */
public class ContentConvolutionMatcher implements VisualModifier {

  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private boolean balanced = true;
  private int patternSize = 32 * 32 * 3;
  private PoolingLayer.PoolingMode poolingMode = PoolingLayer.PoolingMode.Max;

  /**
   * Gets max value.
   *
   * @return the max value
   */
  public int getMaxValue() {
    return maxValue;
  }

  /**
   * Sets max value.
   *
   * @param maxValue the max value
   * @return the max value
   */
  @Nonnull
  public ContentConvolutionMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  /**
   * Gets min value.
   *
   * @return the min value
   */
  public int getMinValue() {
    return minValue;
  }

  /**
   * Sets min value.
   *
   * @param minValue the min value
   * @return the min value
   */
  @Nonnull
  public ContentConvolutionMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  /**
   * Gets pattern size.
   *
   * @return the pattern size
   */
  public int getPatternSize() {
    return patternSize;
  }

  /**
   * Sets pattern size.
   *
   * @param patternSize the pattern size
   * @return the pattern size
   */
  @Nonnull
  public ContentConvolutionMatcher setPatternSize(int patternSize) {
    this.patternSize = patternSize;
    return this;
  }

  /**
   * Gets pooling mode.
   *
   * @return the pooling mode
   */
  public PoolingLayer.PoolingMode getPoolingMode() {
    return poolingMode;
  }

  /**
   * Sets pooling mode.
   *
   * @param poolingMode the pooling mode
   * @return the pooling mode
   */
  @Nonnull
  public ContentConvolutionMatcher setPoolingMode(PoolingLayer.PoolingMode poolingMode) {
    this.poolingMode = poolingMode;
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
  public ContentConvolutionMatcher setAveraging(boolean averaging) {
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
  public ContentConvolutionMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    Tensor baseContent = Result.getData0(network.eval(visualModifierParameters.getStyle()));
    double mag = balanced ? baseContent.rms() : 1;
    int[] baseContentDimensions = baseContent.getDimensions();
    int patternSize = (int) Math.ceil(Math.sqrt(getPatternSize() / baseContentDimensions[2]));
    PoolingLayer poolingLayer = new PoolingLayer();
    poolingLayer.setMode(getPoolingMode());
    poolingLayer.setStrideXY((int) Math.max(1, Math.floor((double) baseContentDimensions[0] / patternSize)), (int) Math.max(1, Math.floor((double) baseContentDimensions[1] / patternSize)));
    poolingLayer.setWindowXY((int) Math.max(1, Math.floor((double) baseContentDimensions[0] / patternSize)), (int) Math.max(1, Math.floor((double) baseContentDimensions[1] / patternSize)));
    Tensor pooledContent = Result.getData0(poolingLayer.eval(baseContent));
    network.add(poolingLayer).freeRef();
    int[] pooledContentDimensions = pooledContent.getDimensions();
    pooledContent.scaleInPlace(Math.pow(pooledContent.rms(), -2));
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(pooledContentDimensions[0], pooledContentDimensions[1], pooledContentDimensions[2], 1);
    convolutionLayer.setPaddingXY(0, 0);
    convolutionLayer.set(pooledContent.permuteDimensions(Integer.MAX_VALUE, -1, 2));
    pooledContent.freeRef();
    Layer explode = convolutionLayer.explode();
    convolutionLayer.freeRef();
    network.add(explode).freeRef();
    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(getMinValue());
    boundedActivationLayer1.setMaxValue(getMaxValue());
    final Layer[] layers = new Layer[]{
        boundedActivationLayer1, new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer()};
    Layer layer = PipelineNetwork.build(1, layers);
    layer.setName(RefString.format("-RMS / %.0E", mag));
    network.add(layer).freeRef();

    {
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      linearActivationLayer.setScale(visualModifierParameters.scale);
      linearActivationLayer.freeze();
      network.add(linearActivationLayer).freeRef();
    }
    visualModifierParameters.freeRef();

    network.freeze();
    return network;
  }
}
