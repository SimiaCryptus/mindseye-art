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

import com.simiacryptus.mindseye.art.TiledTrainable;
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;

public class MomentMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(MomentMatcher.class);
  private static int padding = 8;
  private Precision precision = Precision.Float;
  private int tileSize = 800;
  private double posCoeff = 1.0;
  private double scaleCoeff = 1.0;
  private double covCoeff = 1.0;

  public double getCovCoeff() {
    return covCoeff;
  }

  @Nonnull
  public MomentMatcher setCovCoeff(double covCoeff) {
    this.covCoeff = covCoeff;
    return this;
  }

  public double getPosCoeff() {
    return posCoeff;
  }

  @Nonnull
  public MomentMatcher setPosCoeff(double posCoeff) {
    this.posCoeff = posCoeff;
    return this;
  }

  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  public MomentMatcher setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }

  public double getScaleCoeff() {
    return scaleCoeff;
  }

  @Nonnull
  public MomentMatcher setScaleCoeff(double scaleCoeff) {
    this.scaleCoeff = scaleCoeff;
    return this;
  }

  public int getTileSize() {
    return tileSize;
  }

  @Nonnull
  public MomentMatcher setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  @Nonnull
  public static Layer lossSq(Precision precision, @Nonnull Tensor target) {
    double rms = target.rms();
    final Tensor bias = target.scale(0 == rms ? 1 : -Math.pow(rms, -1));
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = 0 == rms ? 1 : Math.pow(rms, -1);
    linearActivationLayer.setScale(scale);
    MultiPrecision avgReducerLayerMultiPrecision = new AvgReducerLayer();
    avgReducerLayerMultiPrecision.setPrecision(precision);
    MultiPrecision squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(precision);
    MultiPrecision imgBandBiasLayerMultiPrecision = new ImgBandBiasLayer(bias);
    imgBandBiasLayerMultiPrecision.setPrecision(precision);
    final Layer[] layers = new Layer[]{linearActivationLayer.addRef(),
        (ImgBandBiasLayer) RefUtil.addRef(imgBandBiasLayerMultiPrecision), (SquareActivationLayer) RefUtil.addRef(squareActivationLayerMultiPrecision),
        (AvgReducerLayer) RefUtil.addRef(avgReducerLayerMultiPrecision)};
    Layer layer = PipelineNetwork.build(1, layers);
    final String name = RefString.format("RMS[x-C] / %.0E", 0 == rms ? 1 : rms);
    layer.setName(name);
    return layer.addRef();
  }

  @Nonnull
  public static Tensor sum(@Nonnull RefStream<Tensor> tensorStream) {
    return tensorStream.reduce((a, b) -> {
      return Tensor.add(a,b);
    }).orElse(null);
  }

  @Nonnull
  public static UUID getAppendUUID(@Nonnull PipelineNetwork network, @Nonnull Class<GramianLayer> layerClass) {
    DAGNode head = network.getHead();
    Layer layer = head.getLayer();
    head.freeRef();
    if (null == layer)
      return UUID.randomUUID();
    return UUID.nameUUIDFromBytes((layer.getId().toString() + layerClass.getName()).getBytes());
  }

  @Nonnull
  public static Tensor transform(PipelineNetwork network, Tensor in, Precision precision) {
    network = network.copyPipeline();
    MultiPrecision.setPrecision(network, precision);
    assert network != null;
    network.visitLayers(layer -> {
      if (layer instanceof ImgBandBiasLayer) {
        ((ImgBandBiasLayer) layer).setWeights(i -> 0);
      } else {
        //log.info(String.format("Layer %s: %s", layer.getClass().getSimpleName(), layer.getName()));
      }
    });
    final Tensor tensor = network.eval(in).getData().get(0);
    network.freeRef();
    return tensor;
  }

  @Nonnull
  public static PipelineNetwork gateNetwork(@Nonnull PipelineNetwork network, Tensor finalMask) {
    final PipelineNetwork copyPipeline = network.copyPipeline();
    assert copyPipeline != null;
    final DAGNode head = copyPipeline.getHead();
    copyPipeline.add(new ProductLayer(), head, copyPipeline.constValue(finalMask)).freeRef();
    return copyPipeline;
  }

  @Nonnull
  public static Tensor toMask(@Nonnull Tensor tensor) {
    return tensor.mapPixels(pixel -> {
      if (RefArrays.stream(pixel).filter(x -> x != 0).findFirst().isPresent()) {
        return RefArrays.stream(pixel).map(x -> 1).toArray();
      } else {
        return RefArrays.stream(pixel).map(x -> 0).toArray();
      }
    });
  }

  public static boolean test(@Nonnull PipelineNetwork network, @Nonnull Tensor... images) {
    if (images.length > 1)
      return RefUtil.get(RefArrays.stream(images).map(x -> test(network, x)).reduce((a, b) -> a && b));
    try {
      network.eval(images[0]).getData().freeRef();
      return true;
    } catch (Throwable e) {
      throw new RuntimeException(e);
      //return false;
    }
  }

  @Nonnull
  protected static Tensor eval(int pixels, @Nonnull PipelineNetwork network, int tileSize, double power, @Nonnull Tensor[] image) {
    if (image.length <= 0) {
      throw new IllegalArgumentException("image.length <= 0");
    }
    final Tensor sum = sum(RefArrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      final Layer[] selectors = TiledTrainable.selectors(padding, imageDimensions[0], imageDimensions[1], tileSize,
          true);
      if (selectors.length <= 0) {
        throw new IllegalArgumentException("selectors.length <= 0");
      }
      return RefArrays.stream(selectors).map(selector -> {
        //log.info(selector.toString());
        Tensor tile = selector.eval(img).getData().get(0);
        selector.freeRef();
        int[] tileDimensions = tile.getDimensions();
        int tilePixels = tileDimensions[0] * tileDimensions[1];
        Tensor tensor = network.eval(tile).getData().get(0).map(x -> Math.pow(x, power));
        tensor.scaleInPlace(tilePixels);
        Tensor component = tensor.addRef();
        tile.freeRef();
        return component;
      });
    }));
    sum.scaleInPlace(1.0 / pixels);
    return sum.addRef().map(x -> Math.pow(x, 0 == power ? 1 : 1.0 / power)).map(x -> {
      if (Double.isFinite(x)) {
        return x;
      } else {
        return 0;
      }
    });
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.network;
    assert network != null;
    network = network.copyPipeline();
    MultiPrecision.setPrecision(network, precision);

    assert network != null;
    Tensor evalRoot = avg(network, getPixels(visualModifierParameters.style), visualModifierParameters.style);
    assert evalRoot != null;
    double sumSq = evalRoot.sumSq();
    evalRoot.freeRef();
    log.info(RefString.format("Adjust for %s : %s", network.getName(), sumSq));
    final Layer nextHead = new ScaleLayer(0 == sumSq ? 1 : 1.0 / sumSq);
    network.add(nextHead).freeRef();

    final Tensor boolMask = toMask(transform(network, visualModifierParameters.mask, getPrecision()));
    log.info("Mask: " + RefArrays.toString(boolMask.getDimensions()));
    final double maskFactor = RefArrays.stream(boolMask.getData()).average().getAsDouble();
    PipelineNetwork maskedNetwork = network.copyPipeline();
    MultiPrecision.setPrecision(maskedNetwork, getPrecision());
    assert maskedNetwork != null;
    assert test(maskedNetwork, visualModifierParameters.mask);
    final MomentParams params = getMomentParams(network, maskFactor, visualModifierParameters.style);
    network.freeRef();
    assert test(maskedNetwork, visualModifierParameters.mask);
    final DAGNode head = maskedNetwork.getHead();
    maskedNetwork.add(new ProductLayer(), head, maskedNetwork.constValue(boolMask)).freeRef();
    assert test(maskedNetwork, visualModifierParameters.mask);
    final MomentParams nodes = getMomentNodes(maskedNetwork, maskFactor);
    assert test(maskedNetwork, visualModifierParameters.mask);
    final MomentParams momentParams = new MomentParams(nodes.avgNode.addRef(), params.avgValue.addRef(),
        nodes.rmsNode.addRef(), params.rmsValue.addRef(), nodes.covNode.addRef(), params.covValue.addRef(),
        MomentMatcher.this);
    params.freeRef();
    nodes.freeRef();
    momentParams.addLoss(maskedNetwork).freeRef();
    assert test(maskedNetwork, visualModifierParameters.mask);
    visualModifierParameters.freeRef();
    MultiPrecision.setPrecision(maskedNetwork, getPrecision());
    maskedNetwork.freeze();
    return maskedNetwork.addRef();
  }

  public int getPixels(@Nonnull Tensor[] images) {
    return Math.max(1, RefArrays.stream(images).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum());
  }

  @Nonnull
  public MomentMatcher.MomentParams getMomentParams(@Nonnull PipelineNetwork network, double maskFactor, @Nonnull Tensor... images) {
    int pixels = getPixels(images);
    DAGNode mainIn = network.getHead();
    MultiPrecision bandAvgReducerLayerMultiPrecision1 = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode avgNode = network.add((BandAvgReducerLayer) RefUtil.addRef(bandAvgReducerLayerMultiPrecision1), mainIn.addRef()); // Scale the average metrics by 1/x
    Tensor avgValue = eval(pixels, network, getTileSize(), 1.0, images);

    MultiPrecision scaleLayerMultiPrecision2 = new ScaleLayer(-1 / maskFactor);
    scaleLayerMultiPrecision2.setPrecision(getPrecision());
    MultiPrecision imgBandDynamicBiasLayerMultiPrecision = new ImgBandDynamicBiasLayer();
    imgBandDynamicBiasLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode recentered = network.add((ImgBandDynamicBiasLayer) RefUtil.addRef(imgBandDynamicBiasLayerMultiPrecision), mainIn,
        network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision2), avgNode.addRef()));
    NthPowerActivationLayer nthPowerActivationLayer1 = new NthPowerActivationLayer();
    nthPowerActivationLayer1.setPower(0.5);
    MultiPrecision squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision bandAvgReducerLayerMultiPrecision = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision scaleLayerMultiPrecision1 = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode rmsNode = network.add(nthPowerActivationLayer1.addRef(),
        network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision1),
            network.add((BandAvgReducerLayer) RefUtil.addRef(bandAvgReducerLayerMultiPrecision),
                network.add((SquareActivationLayer) RefUtil.addRef(squareActivationLayerMultiPrecision), recentered.addRef()))));
    Tensor rmsValue = eval(pixels, network, getTileSize(), 2.0, images);

    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(0.0);
    BoundedActivationLayer boundedActivationLayer = boundedActivationLayer1.addRef();
    boundedActivationLayer.setMaxValue(1e4);
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-1);
    MultiPrecision productLayerMultiPrecision = new ProductLayer();
    productLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode rescaled = network.add((ProductLayer) RefUtil.addRef(productLayerMultiPrecision), recentered,
        network.add(boundedActivationLayer.addRef(),
            network.add(nthPowerActivationLayer.addRef(), rmsNode.addRef())));

    MultiPrecision gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network, GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision scaleLayerMultiPrecision = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode covNode = network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision), network
        .add((GramianLayer) RefUtil.addRef(gramianLayerMultiPrecision), rescaled)); // Scale the gram matrix by 1/x (elements are averages)
    Tensor covValue = eval(pixels, network, getTileSize(), 1.0, images);

    return new MomentParams(avgNode, avgValue, rmsNode, rmsValue, covNode, covValue, MomentMatcher.this);
  }

  @Nonnull
  public MomentMatcher.MomentParams getMomentNodes(@Nonnull PipelineNetwork network, double maskFactor) {
    DAGNode mainIn = network.getHead();
    MultiPrecision bandAvgReducerLayerMultiPrecision1 = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode avgNode = network.add((BandAvgReducerLayer) RefUtil.addRef(bandAvgReducerLayerMultiPrecision1), mainIn.addRef()); // Scale the average metrics by 1/x

    MultiPrecision scaleLayerMultiPrecision2 = new ScaleLayer(-1 / maskFactor);
    scaleLayerMultiPrecision2.setPrecision(getPrecision());
    MultiPrecision imgBandDynamicBiasLayerMultiPrecision = new ImgBandDynamicBiasLayer();
    imgBandDynamicBiasLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode recentered = network.add((ImgBandDynamicBiasLayer) RefUtil.addRef(imgBandDynamicBiasLayerMultiPrecision), mainIn,
        network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision2), avgNode.addRef()));
    NthPowerActivationLayer nthPowerActivationLayer1 = new NthPowerActivationLayer();
    nthPowerActivationLayer1.setPower(0.5);
    MultiPrecision squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision bandAvgReducerLayerMultiPrecision = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision scaleLayerMultiPrecision1 = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode rmsNode = network.add(nthPowerActivationLayer1.addRef(),
        network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision1),
            network.add((BandAvgReducerLayer) RefUtil.addRef(bandAvgReducerLayerMultiPrecision),
                network.add((SquareActivationLayer) RefUtil.addRef(squareActivationLayerMultiPrecision), recentered.addRef()))));

    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(0.0);
    BoundedActivationLayer boundedActivationLayer = boundedActivationLayer1.addRef();
    boundedActivationLayer.setMaxValue(1e4);
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-1);
    MultiPrecision productLayerMultiPrecision = new ProductLayer();
    productLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode rescaled = network.add((ProductLayer) RefUtil.addRef(productLayerMultiPrecision), recentered,
        network.add(boundedActivationLayer.addRef(),
            network.add(nthPowerActivationLayer.addRef(), rmsNode.addRef())));

    MultiPrecision gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network, GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(getPrecision());
    MultiPrecision scaleLayerMultiPrecision = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode covNode = network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision), network
        .add((GramianLayer) RefUtil.addRef(gramianLayerMultiPrecision), rescaled)); // Scale the gram matrix by 1/x (elements are averages)

    return new MomentParams(avgNode, null, rmsNode, null, covNode, null, MomentMatcher.this);
  }

  @Nullable
  protected Tensor avg(@Nonnull PipelineNetwork network, int pixels, @Nonnull Tensor[] image) {
    //    return eval(pixels, network, getTileSize(), 1.0, image);
    PipelineNetwork avgNet = PipelineNetwork.build(1, network.addRef(), new BandAvgReducerLayer());
    Tensor evalRoot = eval(pixels, avgNet, getTileSize(), 1.0, image);
    avgNet.freeRef();
    return evalRoot;
  }

  private static class MomentParams extends ReferenceCountingBase {
    private final InnerNode avgNode;
    private final Tensor avgValue;
    private final InnerNode rmsNode;
    private final Tensor rmsValue;
    private final InnerNode covNode;
    private final Tensor covValue;
    private final MomentMatcher parent;

    private MomentParams(InnerNode avgNode, Tensor avgValue, InnerNode rmsNode, Tensor rmsValue, InnerNode covNode,
                         Tensor covValue, MomentMatcher parent) {
      this.parent = parent;
      this.avgNode = avgNode;
      this.avgValue = avgValue;
      this.rmsNode = rmsNode;
      this.rmsValue = rmsValue;
      this.covNode = covNode;
      this.covValue = covValue;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    MomentParams[] addRefs(@Nullable MomentParams[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter(x -> x != null).map(momentParams -> momentParams.addRef())
          .toArray(x -> new MomentParams[x]);
    }

    public void _free() {
      if (null != avgNode)
        avgNode.freeRef();
      if (null != avgValue)
        avgValue.freeRef();
      if (null != rmsNode)
        rmsNode.freeRef();
      if (null != rmsValue)
        rmsValue.freeRef();
      if (null != covNode)
        covNode.freeRef();
      if (null != covValue)
        covValue.freeRef();
      super._free();
    }

    @Nullable
    public InnerNode addLoss(@Nonnull PipelineNetwork network) {
      MultiPrecision scaleLayerMultiPrecision = new ScaleLayer(parent.getCovCoeff());
      scaleLayerMultiPrecision.setPrecision(parent.getPrecision());
      MultiPrecision scaleLayerMultiPrecision1 = new ScaleLayer(parent.getScaleCoeff());
      scaleLayerMultiPrecision1.setPrecision(parent.getPrecision());
      MultiPrecision scaleLayerMultiPrecision2 = new ScaleLayer(parent.getPosCoeff());
      scaleLayerMultiPrecision2.setPrecision(parent.getPrecision());
      MultiPrecision sumInputsLayerMultiPrecision = new SumInputsLayer();
      sumInputsLayerMultiPrecision.setPrecision(parent.getPrecision());
      final InnerNode wrap = network.add((SumInputsLayer) RefUtil.addRef(sumInputsLayerMultiPrecision),
          network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision2),
              network.add(lossSq(parent.getPrecision(), avgValue), avgNode.addRef())),
          network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision1),
              network.add(lossSq(parent.getPrecision(), rmsValue), rmsNode.addRef())),
          network.add((ScaleLayer) RefUtil.addRef(scaleLayerMultiPrecision),
              network.add(lossSq(parent.getPrecision(), covValue), covNode.addRef())));
      freeRef();
      return wrap;
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    MomentParams addRef() {
      return (MomentParams) super.addRef();
    }
  }
}
