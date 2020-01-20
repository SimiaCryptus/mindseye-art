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
import com.simiacryptus.mindseye.art.util.PCA;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

public class PatternPCAMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(PatternPCAMatcher.class);
  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private int bands = 16;

  public int getBands() {
    return bands;
  }

  @Nonnull
  public PatternPCAMatcher setBands(int bands) {
    this.bands = bands;
    return this;
  }

  public int getMaxValue() {
    return maxValue;
  }

  @Nonnull
  public PatternPCAMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  public int getMinValue() {
    return minValue;
  }

  @Nonnull
  public PatternPCAMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public PatternPCAMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    assert network != null;
    network = network.copyPipeline();
    assert network != null;
    Tensor baseContent = network.eval(visualModifierParameters.style).getData().get(0);
    visualModifierParameters.freeRef();
    int[] contentDimensions = baseContent.getDimensions();
    RefList<Tensor> components;
    PipelineNetwork signalProjection;
    try {
      PCA pca = new PCA().setRecenter(true).setRescale(false).setEigenvaluePower(0.0);
      Tensor channelMeans = pca.getChannelMeans(baseContent);
      Tensor channelRms = pca.getChannelRms(baseContent, contentDimensions[2], channelMeans);
      assert channelRms != null;
      double[] covariance = PCA.bandCovariance(baseContent.getPixelStream(), PCA.countPixels(baseContent),
          channelMeans.getData(), channelRms.getData());
      channelMeans.scaleInPlace(-1);
      signalProjection = PipelineNetwork.build(1, new ImgBandBiasLayer(channelMeans.addRef()),
          new ImgBandScaleLayer(channelRms.map(x -> 1 / x).getData()));
      channelMeans.freeRef();
      components = PCA.pca(covariance, pca.getEigenvaluePower()).stream().collect(RefCollectors.toList());
    } catch (Throwable e) {
      log.info("Error processing PCA for dimensions " + RefArrays.toString(contentDimensions), e);
      PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
      pipelineNetwork.add(new ValueLayer(new Tensor(0.0)), new DAGNode[]{});
      return pipelineNetwork;
    }
    int bands = Math.min(getBands(), contentDimensions[2]);

    Tensor prefixPattern = signalProjection.eval(baseContent).getData().get(0);
    channelStats(getConvolutionLayer1(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0),
        components, bands).explode().eval(prefixPattern).getData().get(0), bands);
    channelStats(getConvolutionLayer2(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0),
        components, bands).explode().eval(prefixPattern).getData().get(0), bands);
    signalProjection
        .add(getConvolutionLayer1(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0),
            components, bands).explode());
    Tensor spacialPattern = signalProjection.eval(baseContent).getData().get(0);
    channelStats(spacialPattern, bands);

    DAGNode signalProjectionHead = signalProjection.getHead();
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-0.5);
    signalProjection.add(new ProductLayer(), signalProjectionHead, signalProjection.add(PipelineNetwork.build(1,
        new SquareActivationLayer(), new AvgReducerLayer(), nthPowerActivationLayer.addRef())));
    Tensor tensor = spacialPattern.unit();
    tensor.scaleInPlace(-1);
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(contentDimensions[0], contentDimensions[1], bands, 1).setPaddingXY(0, 0);
    convolutionLayer.set(tensor.addRef());
    signalProjection.add(convolutionLayer.addRef().explode());
    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(getMinValue());
    BoundedActivationLayer boundedActivationLayer = boundedActivationLayer1.addRef();
    boundedActivationLayer.setMaxValue(getMaxValue());
    signalProjection.add(boundedActivationLayer.addRef());
    final Layer nextHead = isAveraging() ? new AvgReducerLayer() : new SumReducerLayer();
    signalProjection.add(nextHead);

    signalProjection.setName(RefString.format("PCA Match"));
    network.add(signalProjection.addRef()).freeRef();
    network.freeze();
    return network.addRef();
  }

  public void channelStats(@Nonnull Tensor spacialPattern, int bands) {
    double[] means = RefIntStream.range(0, bands).mapToDouble(band -> {
      return spacialPattern.selectBand(band).mean();
    }).toArray();
    double[] stdDevs = RefIntStream.range(0, bands).mapToDouble(band -> {
      Tensor bandPattern = spacialPattern.selectBand(band);
      return Math.sqrt(Math.pow(bandPattern.rms(), 2) - Math.pow(bandPattern.mean(), 2));
    }).toArray();
    log.info("Means: " + RefArrays.toString(means) + "; StdDev: " + RefArrays.toString(stdDevs));
  }

  @Nonnull
  public ConvolutionLayer getConvolutionLayer1(@Nonnull ConvolutionLayer convolutionLayer, @Nonnull RefList<Tensor> components,
                                               int stride) {
    convolutionLayer.getKernel().setByCoord(c -> {
      int[] coords = c.getCoords();
      return components.get(coords[2] % stride).get(coords[2] / stride);
    });
    return convolutionLayer;
  }

  @Nonnull
  public ConvolutionLayer getConvolutionLayer2(@Nonnull ConvolutionLayer convolutionLayer, @Nonnull RefList<Tensor> components,
                                               int stride) {
    convolutionLayer.getKernel().setByCoord(c -> {
      int[] coords = c.getCoords();
      return components.get(coords[2] / stride).get(coords[2] % stride);
    });
    return convolutionLayer;
  }
}
