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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.ValueLayer;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

/**
 * The type Content pca matcher.
 */
public class ContentPCAMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(ContentPCAMatcher.class);
  private int minValue = -1;
  private int maxValue = 1;
  private boolean averaging = true;
  private int bands = 16;

  /**
   * Gets bands.
   *
   * @return the bands
   */
  public int getBands() {
    return bands;
  }

  /**
   * Sets bands.
   *
   * @param bands the bands
   */
  public void setBands(int bands) {
    this.bands = bands;
  }

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
   */
  public void setMaxValue(int maxValue) {
    this.maxValue = maxValue;
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
   */
  public void setMinValue(int minValue) {
    this.minValue = minValue;
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
   */
  public void setAveraging(boolean averaging) {
    this.averaging = averaging;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    Tensor baseContent = Result.getData0(network.eval(visualModifierParameters.getStyle()));
    int[] contentDimensions = baseContent.getDimensions();
    final RefList<Tensor> components;
    final PipelineNetwork signalProjection;
    try {
      PCA pca = new PCA().setRecenter(true).setRescale(false).setEigenvaluePower(0.0);
      Tensor channelMeans = pca.getChannelMeans(baseContent.addRef());
      Tensor channelRms = pca.getChannelRms(baseContent.addRef(), contentDimensions[2], channelMeans.addRef());
      assert channelRms != null;
      double[] covariance = PCA.bandCovariance(baseContent.getPixelStream(), PCA.countPixels(baseContent.addRef()),
          channelMeans.getData(), channelRms.getData());
      channelMeans.scaleInPlace(-1);
      Tensor map = channelRms.map(x -> 1 / x);
      channelRms.freeRef();
      signalProjection = PipelineNetwork.build(1, new ImgBandBiasLayer(channelMeans),
          new ImgBandScaleLayer(map.getData()));
      map.freeRef();
      RefList<Tensor> pca1 = PCA.pca(covariance, pca.getEigenvaluePower());
      components = pca1.stream().collect(RefCollectors.toList());
      pca1.freeRef();
    } catch (Throwable e) {
      log.info("Error processing PCA for dimensions " + RefArrays.toString(contentDimensions), e);
      PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
      pipelineNetwork.add(new ValueLayer(new Tensor(0.0)), new DAGNode[]{}).freeRef();

      {
        LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
        linearActivationLayer.setScale(visualModifierParameters.scale);
        linearActivationLayer.freeze();
        network.add(linearActivationLayer).freeRef();
      }
      visualModifierParameters.freeRef();

      return pipelineNetwork;
    }
    int bands = Math.min(getBands(), contentDimensions[2]);

    Tensor prefixPattern = Result.getData0(signalProjection.eval(baseContent.addRef()));
    ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(1, 1, contentDimensions[2], bands);
    convolutionLayer2.setPaddingXY(0, 0);
    ConvolutionLayer convolutionLayer12 = getConvolutionLayer1(convolutionLayer2, components.addRef(), bands);
    Layer explode1 = convolutionLayer12.explode();
    convolutionLayer12.freeRef();
    channelStats(Result.getData0(explode1.eval(prefixPattern.addRef())), bands);
    explode1.freeRef();
    ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, contentDimensions[2], bands);
    convolutionLayer1.setPaddingXY(0, 0);
    ConvolutionLayer convolutionLayer21 = getConvolutionLayer2(convolutionLayer1,
        components.addRef(), bands);
    Layer explode = convolutionLayer21.explode();
    convolutionLayer21.freeRef();
    channelStats(Result.getData0(explode.eval(prefixPattern)), bands);
    explode.freeRef();


    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, contentDimensions[2], bands);
    convolutionLayer.setPaddingXY(0, 0);
    ConvolutionLayer convolutionLayer11 = getConvolutionLayer1(convolutionLayer, components, bands);
    signalProjection.add(convolutionLayer11.explode()).freeRef();
    convolutionLayer11.freeRef();
    Tensor spacialPattern = Result.getData0(signalProjection.eval(baseContent));
    channelStats(spacialPattern.addRef(), bands);

    double mag = spacialPattern.rms();
    DAGNode head = signalProjection.getHead();
    spacialPattern.scaleInPlace(-1);
    DAGNode constNode = signalProjection.constValueWrap(spacialPattern);
    Layer layer1 = new SumInputsLayer();
    layer1.setName("Difference");
    signalProjection.add(layer1, head, constNode).freeRef();

    {
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      linearActivationLayer.setScale(Math.pow(mag, -2));
      Layer layer = PipelineNetwork.build(1,
              new SquareActivationLayer(),
              isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
              linearActivationLayer
      );
      layer.setName(RefString.format("RMS / %.0E", mag));
      signalProjection.add(layer).freeRef();
      signalProjection.setName(RefString.format("PCA Content Match"));
      network.add(signalProjection).freeRef();
    }

    {
      LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
      linearActivationLayer.setScale(visualModifierParameters.scale);
      linearActivationLayer.freeze();
      network.add(linearActivationLayer).freeRef();
      visualModifierParameters.freeRef();
    }

    network.freeze();
    return network;
  }

  /**
   * Channel stats.
   *
   * @param spacialPattern the spacial pattern
   * @param bands          the bands
   */
  public void channelStats(@Nonnull Tensor spacialPattern, int bands) {
    double[] means = RefIntStream.range(0, bands).mapToDouble(band -> {
      Tensor selectBand = spacialPattern.selectBand(band);
      double mean = selectBand.mean();
      selectBand.freeRef();
      return mean;
    }).toArray();
    double[] stdDevs = RefIntStream.range(0, bands).mapToDouble(band -> {
      Tensor bandPattern = spacialPattern.selectBand(band);
      double sqrt = Math.sqrt(Math.pow(bandPattern.rms(), 2) - Math.pow(bandPattern.mean(), 2));
      bandPattern.freeRef();
      return sqrt;
    }).toArray();
    spacialPattern.freeRef();
    log.info("Means: " + RefArrays.toString(means) + "; StdDev: " + RefArrays.toString(stdDevs));
  }

  /**
   * Gets convolution layer 1.
   *
   * @param convolutionLayer the convolution layer
   * @param components       the components
   * @param stride           the stride
   * @return the convolution layer 1
   */
  @Nonnull
  public ConvolutionLayer getConvolutionLayer1(@Nonnull ConvolutionLayer convolutionLayer, @Nonnull RefList<Tensor> components,
                                               int stride) {
    convolutionLayer.setByCoord(c -> {
      int[] coords = c.getCoords();
      Tensor tensor = components.get(coords[2] % stride);
      double v = tensor.get(coords[2] / stride);
      tensor.freeRef();
      return v;
    });
    components.freeRef();
    return convolutionLayer;
  }

  /**
   * Gets convolution layer 2.
   *
   * @param convolutionLayer the convolution layer
   * @param components       the components
   * @param stride           the stride
   * @return the convolution layer 2
   */
  @Nonnull
  public ConvolutionLayer getConvolutionLayer2(@Nonnull ConvolutionLayer convolutionLayer, @Nonnull RefList<Tensor> components,
                                               int stride) {
    convolutionLayer.setByCoord(c -> {
      int[] coords = c.getCoords();
      Tensor tensor = components.get(coords[2] / stride);
      double v = tensor.get(coords[2] % stride);
      tensor.freeRef();
      return v;
    });
    components.freeRef();
    return convolutionLayer;
  }
}
