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
import com.simiacryptus.mindseye.art.util.PCA;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ContentPCAMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(ContentPCAMatcher.class);
  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private int bands = 16;

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor... image) {
    network = network.copyPipeline();
    Tensor baseContent = network.eval(image).getDataAndFree().getAndFree(0);
    int[] contentDimensions = baseContent.getDimensions();
    List<Tensor> components;
    PipelineNetwork signalProjection;
    try {
      PCA pca = new PCA().setRecenter(true).setRescale(false).setEigenvaluePower(0.0);
      Tensor channelMeans = pca.getChannelMeans(baseContent);
      Tensor channelRms = pca.getChannelRms(baseContent, contentDimensions[2], channelMeans);
      double[] covariance = PCA.bandCovariance(baseContent.getPixelStream(), PCA.countPixels(baseContent), channelMeans.getData(), channelRms.getData());
      signalProjection = PipelineNetwork.wrap(1,
          new ImgBandBiasLayer(channelMeans.scaleInPlace(-1)),
          new ImgBandScaleLayer(channelRms.mapAndFree(x -> 1 / x).getData())
      );
      channelMeans.freeRef();
      components = PCA.pca(covariance, pca.getEigenvaluePower()).stream().collect(Collectors.toList());
    } catch (Throwable e) {
      log.info("Error processing PCA for dimensions " + Arrays.toString(contentDimensions), e);
      PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
      pipelineNetwork.wrap(new ValueLayer(new Tensor(0.0)), new DAGNode[]{});
      return pipelineNetwork;
    }
    int bands = Math.min(getBands(), contentDimensions[2]);

    Tensor prefixPattern = signalProjection.eval(baseContent).getDataAndFree().getAndFree(0);
    channelStats(getConvolutionLayer1(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0), components, bands).explodeAndFree().eval(prefixPattern).getDataAndFree().getAndFree(0), bands);
    channelStats(getConvolutionLayer2(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0), components, bands).explodeAndFree().eval(prefixPattern).getDataAndFree().getAndFree(0), bands);

    signalProjection.wrap(getConvolutionLayer1(new ConvolutionLayer(1, 1, contentDimensions[2], bands).setPaddingXY(0, 0), components, bands).explodeAndFree());
    Tensor spacialPattern = signalProjection.eval(baseContent).getDataAndFree().getAndFree(0);
    channelStats(spacialPattern, bands);

    double mag = spacialPattern.rms();
    DAGNode head = signalProjection.getHead();
    DAGNode constNode = signalProjection.constValueWrap(spacialPattern.scaleInPlace(-1));
    signalProjection.wrap(new SumInputsLayer().setName("Difference"), head, constNode).freeRef();
    signalProjection.wrap(PipelineNetwork.wrap(1,
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new LinearActivationLayer().setScale(Math.pow(mag, -2))
//        ,new NthPowerActivationLayer().setPower(0.5)
    ).setName(String.format("RMS / %.0E", mag)));

    network.wrap(signalProjection.setName(String.format("PCA Content Match"))).freeRef();
    return (PipelineNetwork) network.freeze();
  }

  public void channelStats(Tensor spacialPattern, int bands) {
    double[] means = IntStream.range(0, bands).mapToDouble(band -> {
      return spacialPattern.selectBand(band).mean();
    }).toArray();
    double[] stdDevs = IntStream.range(0, bands).mapToDouble(band -> {
      Tensor bandPattern = spacialPattern.selectBand(band);
      return Math.sqrt(Math.pow(bandPattern.rms(), 2) - Math.pow(bandPattern.mean(), 2));
    }).toArray();
    log.info("Means: " + Arrays.toString(means) + "; StdDev: " + Arrays.toString(stdDevs));
  }

  @NotNull
  public ConvolutionLayer getConvolutionLayer1(ConvolutionLayer convolutionLayer, List<Tensor> components, int stride) {
    convolutionLayer.getKernel().setByCoord(c -> {
      int[] coords = c.getCoords();
      return components.get(coords[2] % stride).get(coords[2] / stride);
    });
    return convolutionLayer;
  }

  @NotNull
  public ConvolutionLayer getConvolutionLayer2(ConvolutionLayer convolutionLayer, List<Tensor> components, int stride) {
    convolutionLayer.getKernel().setByCoord(c -> {
      int[] coords = c.getCoords();
      return components.get(coords[2] / stride).get(coords[2] % stride);
    });
    return convolutionLayer;
  }

  public boolean isAveraging() {
    return averaging;
  }

  public ContentPCAMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public int getMinValue() {
    return minValue;
  }

  public ContentPCAMatcher setMinValue(int minValue) {
    this.minValue = minValue;
    return this;
  }

  public int getMaxValue() {
    return maxValue;
  }

  public ContentPCAMatcher setMaxValue(int maxValue) {
    this.maxValue = maxValue;
    return this;
  }

  public int getBands() {
    return bands;
  }

  public ContentPCAMatcher setBands(int bands) {
    this.bands = bands;
    return this;
  }
}
