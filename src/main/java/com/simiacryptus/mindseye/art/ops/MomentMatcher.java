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

import com.simiacryptus.mindseye.art.ArtSettings;
import com.simiacryptus.mindseye.art.TiledTrainable;
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
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
import com.simiacryptus.util.Util;
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
  private int tileSize = ArtSettings.INSTANCE().defaultTileSize;
  private double posCoeff = 1.0;
  private double scaleCoeff = 1.0;
  private double covCoeff = 1.0;

  public double getCovCoeff() {
    return covCoeff;
  }

  public void setCovCoeff(double covCoeff) {
    this.covCoeff = covCoeff;
  }

  public double getPosCoeff() {
    return posCoeff;
  }

  public void setPosCoeff(double posCoeff) {
    this.posCoeff = posCoeff;
  }

  public Precision getPrecision() {
    return precision;
  }

  public void setPrecision(Precision precision) {
    this.precision = precision;
  }

  public double getScaleCoeff() {
    return scaleCoeff;
  }

  public void setScaleCoeff(double scaleCoeff) {
    this.scaleCoeff = scaleCoeff;
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
    target.freeRef();
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    final double scale = 0 == rms ? 1 : Math.pow(rms, -1);
    linearActivationLayer.setScale(scale);
    AvgReducerLayer avgReducerLayerMultiPrecision = new AvgReducerLayer();
    avgReducerLayerMultiPrecision.setPrecision(precision);
    SquareActivationLayer squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(precision);
    ImgBandBiasLayer imgBandBiasLayerMultiPrecision = new ImgBandBiasLayer(bias);
    imgBandBiasLayerMultiPrecision.setPrecision(precision);
    Layer layer = PipelineNetwork.build(1,
        linearActivationLayer,
        imgBandBiasLayerMultiPrecision,
        squareActivationLayerMultiPrecision,
        avgReducerLayerMultiPrecision);
    final String name = RefString.format("RMS[x-C] / %.0E", 0 == rms ? 1 : rms);
    layer.setName(name);
    return layer;
  }

  @Nonnull
  public static Tensor sum(@Nonnull RefStream<Tensor> tensorStream) {
    return RefUtil.orElse(tensorStream.reduce((a, b) -> {
      return Tensor.add(a, b);
    }), null);
  }

  @Nonnull
  public static UUID getAppendUUID(@Nonnull PipelineNetwork network, @Nonnull Class<GramianLayer> layerClass) {
    DAGNode head = network.getHead();
    Layer layer = head.getLayer();
    network.freeRef();
    head.freeRef();
    if (null == layer)
      return UUID.randomUUID();
    UUID uuid = UUID.nameUUIDFromBytes((layer.getId().toString() + layerClass.getName()).getBytes());
    layer.freeRef();
    return uuid;
  }

  @Nonnull
  public static Tensor transform(PipelineNetwork network, Tensor in, Precision precision) {
    if (null == in) {
      network.freeRef();
      return null;
    }
    assert in.assertAlive();
    PipelineNetwork copyPipeline = network.copyPipeline();
    network.freeRef();
    MultiPrecision.setPrecision(copyPipeline.addRef(), precision);
    assert copyPipeline != null;
    copyPipeline.visitLayers(layer -> {
      if (layer instanceof ImgBandBiasLayer) {
        ((ImgBandBiasLayer) layer).setWeights(i -> 0);
      } else {
        //log.info(String.format("Layer %s: %s", layer.getClass().getSimpleName(), layer.getName()));
      }
      layer.freeRef();
    });
    Result result = copyPipeline.eval(in);
    TensorList data = result.getData();
    final Tensor tensor = data.get(0);
    result.freeRef();
    data.freeRef();
    copyPipeline.freeRef();
    return tensor;
  }

  @Nonnull
  public static PipelineNetwork gateNetwork(@Nonnull PipelineNetwork network, Tensor finalMask) {
    final PipelineNetwork copyPipeline = network.copyPipeline();
    network.freeRef();
    assert copyPipeline != null;
    final DAGNode head = copyPipeline.getHead();
    copyPipeline.add(new ProductLayer(), head, copyPipeline.constValue(finalMask)).freeRef();
    return copyPipeline;
  }

  @Nonnull
  public static Tensor toMask(@Nonnull Tensor tensor) {
    if (tensor == null) return null;
    Tensor mapPixels = tensor.mapPixels(pixel -> {
      if (Arrays.stream(pixel).filter(x -> x != 0).findFirst().isPresent()) {
        return Arrays.stream(pixel).map(x -> 1).toArray();
      } else {
        return Arrays.stream(pixel).map(x -> 0).toArray();
      }
    });
    tensor.freeRef();
    return mapPixels;
  }

  public static boolean test(@Nonnull PipelineNetwork network, @Nonnull Tensor... images) {
    if (images.length > 1) {
      Boolean test = RefUtil.get(RefArrays.stream(images).map(x -> test(network.addRef(), x)).reduce((a, b) -> a && b));
      network.freeRef();
      return test;
    }
    try {
      network.eval(images[0].addRef()).freeRef();
      network.freeRef();
      RefUtil.freeRef(images);
      return true;
    } catch (Throwable e) {
      throw Util.throwException(e);
      //return false;
    }
  }

  @Nonnull
  protected static Tensor eval(int pixels, @Nonnull PipelineNetwork network, int tileSize, double power, @Nonnull Tensor[] image) {
    if (image.length <= 0) {
      network.freeRef();
      RefUtil.freeRef(image);
      throw new IllegalArgumentException("image.length <= 0");
    }
    final Tensor sum = sum(RefArrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      final Layer[] selectors = TiledTrainable.selectors(padding, imageDimensions[0], imageDimensions[1], tileSize,
          true);
      if (selectors.length <= 0) {
        img.freeRef();
        RefUtil.freeRef(selectors);
        throw new IllegalArgumentException("selectors.length <= 0");
      }
      return RefArrays.stream(selectors).map(RefUtil.wrapInterface(selector -> {
        //log.info(selector.toString());
        Result result = selector.eval(img.addRef());
        selector.freeRef();
        TensorList data = result.getData();
        result.freeRef();
        Tensor tile = data.get(0);
        data.freeRef();
        int[] tileDimensions = tile.getDimensions();
        int tilePixels = tileDimensions[0] * tileDimensions[1];
        Result result1 = network.eval(tile);
        TensorList data1 = result1.getData();
        result1.freeRef();
        Tensor tensor1 = data1.get(0);
        data1.freeRef();
        Tensor tensor = tensor1.map(x -> Math.pow(x, power));
        tensor1.freeRef();
        tensor.scaleInPlace(tilePixels);
        return tensor;
      }, img));
    }));
    network.freeRef();
    sum.scaleInPlace(1.0 / pixels);
    Tensor tensor = sum.map(x -> {
      double x1 = Math.pow(x, 0 == power ? 1 : 1.0 / power);
      if (Double.isFinite(x1)) {
        return x1;
      } else {
        return 0;
      }
    });
    sum.freeRef();
    return tensor;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    PipelineNetwork network = visualModifierParameters.copyNetwork();
    MultiPrecision.setPrecision(network.addRef(), precision);

    assert network != null;
    Tensor evalRoot = avg(network.addRef(), getPixels(visualModifierParameters.getStyle()), visualModifierParameters.getStyle());
    assert evalRoot != null;
    double sumSq = evalRoot.sumSq();
    evalRoot.freeRef();
    log.info(RefString.format("Adjust for %s by %s: %s", network.getName(), this.getClass().getSimpleName(), sumSq));
    double factor;
    if (Double.isFinite(sumSq) && 0 < sumSq) factor = 1;
    else factor = 1.0 / sumSq;
    final Layer nextHead = new ScaleLayer(factor);
    network.add(nextHead).freeRef();

    Tensor mask = visualModifierParameters.getMask();
    final double maskFactor;
    final Tensor boolMask;
    if (null != mask) {
      boolMask = toMask(transform(network.addRef(), mask.addRef(), getPrecision()));
      log.info("Mask: " + RefArrays.toString(boolMask.getDimensions()));
      maskFactor = boolMask.doubleStream().average().getAsDouble();
    } else {
      maskFactor = 1;
      boolMask = null;
    }
    PipelineNetwork maskedNetwork = network.copyPipeline();
    MultiPrecision.setPrecision(maskedNetwork.addRef(), getPrecision());
    assert maskedNetwork != null;
    assert mask == null || test(maskedNetwork.addRef(), mask.addRef());
    final MomentParams params = getMomentParams(network, maskFactor, visualModifierParameters.getStyle());
    assert mask == null || test(maskedNetwork.addRef(), mask.addRef());
    if (null != boolMask) {
      final DAGNode head = maskedNetwork.getHead();
      maskedNetwork.add(new ProductLayer(), head, maskedNetwork.constValue(boolMask)).freeRef();
    }
    assert mask == null || test(maskedNetwork.addRef(), mask.addRef());
    final MomentParams nodes = getMomentNodes(maskedNetwork.addRef(), maskFactor);
    assert mask == null || test(maskedNetwork.addRef(), mask.addRef());
    final MomentParams momentParams = new MomentParams(nodes.avgNode.addRef(), params.avgValue.addRef(),
        nodes.rmsNode.addRef(), params.rmsValue.addRef(), nodes.covNode.addRef(), params.covValue.addRef(),
        MomentMatcher.this);
    params.freeRef();
    nodes.freeRef();
    momentParams.addLoss(maskedNetwork.addRef()).freeRef();
    momentParams.freeRef();
    assert mask == null || test(maskedNetwork.addRef(), mask.addRef());
    visualModifierParameters.freeRef();
    MultiPrecision.setPrecision(maskedNetwork.addRef(), getPrecision());
    maskedNetwork.freeze();
    RefUtil.freeRef(mask);
    return maskedNetwork;
  }

  public int getPixels(@Nonnull Tensor[] images) {
    return Math.max(1, RefArrays.stream(images).mapToInt(tensor -> {
      int[] dimensions = tensor.getDimensions();
      tensor.freeRef();
      return dimensions[0] * dimensions[1];
    }).sum());
  }

  @Nonnull
  public MomentMatcher.MomentParams getMomentParams(@Nonnull PipelineNetwork network, double maskFactor, @Nonnull Tensor... images) {
    int pixels = getPixels(RefUtil.addRef(images));
    DAGNode mainIn = network.getHead();
    BandAvgReducerLayer bandAvgReducerLayerMultiPrecision1 = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode avgNode = network.add(bandAvgReducerLayerMultiPrecision1, mainIn.addRef()); // Scale the average metrics by 1/x
    Tensor avgValue = eval(pixels, network.addRef(), getTileSize(), 1.0, RefUtil.addRef(images));

    ScaleLayer scaleLayerMultiPrecision2 = new ScaleLayer(-1 / maskFactor);
    scaleLayerMultiPrecision2.setPrecision(getPrecision());
    ImgBandDynamicBiasLayer imgBandDynamicBiasLayerMultiPrecision = new ImgBandDynamicBiasLayer();
    imgBandDynamicBiasLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode recentered = network.add(imgBandDynamicBiasLayerMultiPrecision, mainIn,
        network.add(scaleLayerMultiPrecision2, avgNode.addRef()));
    NthPowerActivationLayer nthPowerActivationLayer1 = new NthPowerActivationLayer();
    nthPowerActivationLayer1.setPower(0.5);
    SquareActivationLayer squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(getPrecision());
    BandAvgReducerLayer bandAvgReducerLayerMultiPrecision = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision.setPrecision(getPrecision());
    ScaleLayer scaleLayerMultiPrecision1 = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode rmsNode = network.add(nthPowerActivationLayer1,
        network.add(scaleLayerMultiPrecision1,
            network.add(bandAvgReducerLayerMultiPrecision,
                network.add(squareActivationLayerMultiPrecision, recentered.addRef()))));
    Tensor rmsValue = eval(pixels, network.addRef(), getTileSize(), 2.0, RefUtil.addRef(images));

    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(0.0);
    boundedActivationLayer1.setMaxValue(1e4);
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-1);
    ProductLayer productLayerMultiPrecision = new ProductLayer();
    productLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode rescaled = network.add(productLayerMultiPrecision, recentered,
        network.add(boundedActivationLayer1,
            network.add(nthPowerActivationLayer, rmsNode.addRef())));

    GramianLayer gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network.addRef(), GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(getPrecision());
    ScaleLayer scaleLayerMultiPrecision = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode covNode = network.add(scaleLayerMultiPrecision, network
        .add(gramianLayerMultiPrecision, rescaled)); // Scale the gram matrix by 1/x (elements are averages)
    Tensor covValue = eval(pixels, network, getTileSize(), 1.0, images);
    return new MomentParams(avgNode, avgValue, rmsNode, rmsValue, covNode, covValue, MomentMatcher.this);
  }

  @Nonnull
  public MomentMatcher.MomentParams getMomentNodes(@Nonnull PipelineNetwork network, double maskFactor) {
    DAGNode mainIn = network.getHead();
    BandAvgReducerLayer bandAvgReducerLayerMultiPrecision1 = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode avgNode = network.add(bandAvgReducerLayerMultiPrecision1, mainIn.addRef()); // Scale the average metrics by 1/x

    ScaleLayer scaleLayerMultiPrecision2 = new ScaleLayer(-1 / maskFactor);
    scaleLayerMultiPrecision2.setPrecision(getPrecision());
    ImgBandDynamicBiasLayer imgBandDynamicBiasLayerMultiPrecision = new ImgBandDynamicBiasLayer();
    imgBandDynamicBiasLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode recentered = network.add(imgBandDynamicBiasLayerMultiPrecision, mainIn,
        network.add(scaleLayerMultiPrecision2, avgNode.addRef()));
    NthPowerActivationLayer nthPowerActivationLayer1 = new NthPowerActivationLayer();
    nthPowerActivationLayer1.setPower(0.5);
    SquareActivationLayer squareActivationLayerMultiPrecision = new SquareActivationLayer();
    squareActivationLayerMultiPrecision.setPrecision(getPrecision());
    BandAvgReducerLayer bandAvgReducerLayerMultiPrecision = new BandAvgReducerLayer();
    bandAvgReducerLayerMultiPrecision.setPrecision(getPrecision());
    ScaleLayer scaleLayerMultiPrecision1 = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision1.setPrecision(getPrecision());
    InnerNode rmsNode = network.add(nthPowerActivationLayer1,
        network.add(scaleLayerMultiPrecision1,
            network.add(bandAvgReducerLayerMultiPrecision,
                network.add(squareActivationLayerMultiPrecision, recentered.addRef()))));

    BoundedActivationLayer boundedActivationLayer1 = new BoundedActivationLayer();
    boundedActivationLayer1.setMinValue(0.0);
    boundedActivationLayer1.setMaxValue(1e4);
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-1);
    ProductLayer productLayerMultiPrecision = new ProductLayer();
    productLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode rescaled = network.add(productLayerMultiPrecision, recentered,
        network.add(boundedActivationLayer1,
            network.add(nthPowerActivationLayer, rmsNode.addRef())));

    GramianLayer gramianLayerMultiPrecision = new GramianLayer(getAppendUUID(network.addRef(), GramianLayer.class));
    gramianLayerMultiPrecision.setPrecision(getPrecision());
    ScaleLayer scaleLayerMultiPrecision = new ScaleLayer(1 / maskFactor);
    scaleLayerMultiPrecision.setPrecision(getPrecision());
    InnerNode covNode = network.add(scaleLayerMultiPrecision, network
        .add(gramianLayerMultiPrecision, rescaled)); // Scale the gram matrix by 1/x (elements are averages)
    network.freeRef();
    return new MomentParams(avgNode, null, rmsNode, null, covNode, null, MomentMatcher.this);
  }

  @Nullable
  protected Tensor avg(@Nonnull PipelineNetwork network, int pixels, @Nonnull Tensor[] image) {
    //    return eval(pixels, network, getTileSize(), 1.0, image);
    PipelineNetwork avgNet = PipelineNetwork.build(1, network, new BandAvgReducerLayer());
    return eval(pixels, avgNet, getTileSize(), 1.0, image);
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
      ScaleLayer scaleLayerMultiPrecision = new ScaleLayer(parent.getCovCoeff());
      scaleLayerMultiPrecision.setPrecision(parent.getPrecision());
      ScaleLayer scaleLayerMultiPrecision1 = new ScaleLayer(parent.getScaleCoeff());
      scaleLayerMultiPrecision1.setPrecision(parent.getPrecision());
      ScaleLayer scaleLayerMultiPrecision2 = new ScaleLayer(parent.getPosCoeff());
      scaleLayerMultiPrecision2.setPrecision(parent.getPrecision());
      SumInputsLayer sumInputsLayerMultiPrecision = new SumInputsLayer();
      sumInputsLayerMultiPrecision.setPrecision(parent.getPrecision());
      final InnerNode wrap = network.add(sumInputsLayerMultiPrecision,
          network.add(scaleLayerMultiPrecision2,
              network.add(lossSq(parent.getPrecision(), avgValue.addRef()), avgNode.addRef())),
          network.add(scaleLayerMultiPrecision1,
              network.add(lossSq(parent.getPrecision(), rmsValue.addRef()), rmsNode.addRef())),
          network.add(scaleLayerMultiPrecision,
              network.add(lossSq(parent.getPrecision(), covValue.addRef()), covNode.addRef())));
      network.freeRef();
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
