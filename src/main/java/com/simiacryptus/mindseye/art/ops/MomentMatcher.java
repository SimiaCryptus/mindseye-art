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
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.UUID;
import java.util.stream.Stream;

public class MomentMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(MomentMatcher.class);
  private static int padding = 8;
  private Precision precision = Precision.Float;
  private int tileSize = 400;
  private double posCoeff = 1.0;
  private double scaleCoeff = 1.0;
  private double covCoeff = 1.0;

  @NotNull
  public static Layer lossSq(Precision precision, Tensor target) {
    double rms = target.rms();
    Layer layer = PipelineNetwork.wrap(1,
        new LinearActivationLayer().setScale(0 == rms ? 1 : Math.pow(rms, -1)),
        new ImgBandBiasLayer(target.scaleInPlace(0 == rms ? 1 : -Math.pow(rms, -1))).setPrecision(precision),
        new SquareActivationLayer().setPrecision(precision),
        new AvgReducerLayer().setPrecision(precision)
    ).setName(String.format("RMS[x-C] / %.0E", 0 == rms ? 1 : rms));
    target.freeRef();
    return layer;
  }

  protected static Tensor eval(int pixels, PipelineNetwork network, int tileSize, double power, Tensor[] image) {
    return sum(Arrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      return Arrays.stream(TiledTrainable.selectors(padding, imageDimensions[0], imageDimensions[1], tileSize, true))
          .map(selector -> {
            //log.info(selector.toString());
            Tensor tile = selector.eval(img).getDataAndFree().getAndFree(0);
            selector.freeRef();
            int[] tileDimensions = tile.getDimensions();
            int tilePixels = tileDimensions[0] * tileDimensions[1];
            Tensor component = network.eval(tile).getDataAndFree().getAndFree(0)
                .mapAndFree(x -> Math.pow(x, power))
                .scaleInPlace(tilePixels);
            tile.freeRef();
            return component;
          });
    })).scaleInPlace(1.0 / pixels)
        .mapAndFree(x -> Math.pow(x, 0 == power ? 1 : (1.0 / power)))
        .mapAndFree(x -> {
          if (Double.isFinite(x)) {
            return x;
          } else {
            return 0;
          }
        });
  }


  @NotNull
  public static Tensor sum(Stream<Tensor> tensorStream) {
    return tensorStream.reduce((a, b) -> {
      a.addAndFree(b);
      b.freeRef();
      return a;
    }).get();
  }

  @NotNull
  public static UUID getAppendUUID(PipelineNetwork network, Class<GramianLayer> layerClass) {
    DAGNode head = network.getHead();
    Layer layer = head.getLayer();
    if (null == layer) return UUID.randomUUID();
    return UUID.nameUUIDFromBytes((layer.getId().toString() + layerClass.getName()).getBytes());
  }

  public static Tensor transform(PipelineNetwork network, Tensor in, Precision precision) {
    network = MultiPrecision.setPrecision(network.copyPipeline(), precision);
    network.visitLayers(layer -> {
      if (layer instanceof ImgBandBiasLayer) {
        ((ImgBandBiasLayer) layer).setWeights(i -> 0);
      } else {
        //log.info(String.format("Layer %s: %s", layer.getClass().getSimpleName(), layer.getName()));
      }
    });
    return network.eval(in).getDataAndFree().getAndFree(0);
  }

  @NotNull
  public static PipelineNetwork gateNetwork(PipelineNetwork network, Tensor finalMask) {
    final PipelineNetwork copyPipeline = network.copyPipeline();
    final DAGNode head = copyPipeline.getHead();
    copyPipeline.wrap(new ProductLayer(), head, copyPipeline.constValue(finalMask)).freeRef();
    return copyPipeline;
  }

  public static Tensor toMask(Tensor tensor) {
    return tensor.mapPixelsAndFree(pixel -> {
      if (Arrays.stream(pixel).filter(x -> x != 0).findFirst().isPresent()) {
        return Arrays.stream(pixel).map(x -> 1).toArray();
      } else {
        return Arrays.stream(pixel).map(x -> 0).toArray();
      }
    });
  }

  public static boolean test(PipelineNetwork network, Tensor... images) {
    if (images.length > 1) return Arrays.stream(images).map(x -> test(network, x)).reduce((a, b) -> a && b).get();
    try {
      network.eval(images[0]).getDataAndFree().freeRef();
      return true;
    } catch (Throwable e) {
      throw new RuntimeException(e);
      //return false;
    }
  }

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor mask, Tensor... images) {
    network = MultiPrecision.setPrecision(network.copyPipeline(), getPrecision());

    Tensor evalRoot = avg(network, getPixels(images), images);
    double sumSq = evalRoot.sumSq();
    //evalRoot.freeRef();
    log.info(String.format("Adjust for %s : %s", network.getName(), sumSq));
    network.wrap(new ScaleLayer(0 == sumSq ? 1 : 1.0 / sumSq));

    if (null != mask) {
      final Tensor boolMask = toMask(transform(network, mask, getPrecision()));
      log.info("Mask: " + Arrays.toString(boolMask.getDimensions()));
      final double maskFactor = Arrays.stream(boolMask.getData()).average().getAsDouble();
      PipelineNetwork maskedNetwork = MultiPrecision.setPrecision(network.copyPipeline(), getPrecision());
      assert test(maskedNetwork, mask);
      final MomentParams params = getMomentParams(network, maskFactor, images);
      assert test(maskedNetwork, mask);
      final DAGNode head = maskedNetwork.getHead();
      maskedNetwork.wrap(new ProductLayer(), head, maskedNetwork.constValue(boolMask)).freeRef();
      assert test(maskedNetwork, mask);
      final MomentParams nodes = getMomentNodes(maskedNetwork, maskFactor);
      assert test(maskedNetwork, mask);
      new MomentParams(this,
          nodes.avgNode, params.avgValue,
          nodes.rmsNode, params.rmsValue,
          nodes.covNode, params.covValue
      ).addLoss(maskedNetwork);
      assert test(maskedNetwork, mask);
      MultiPrecision.setPrecision(maskedNetwork, getPrecision());
      return (PipelineNetwork) maskedNetwork.freeze();
    } else {
      getMomentParams(network, 1.0, images).addLoss(network);
      MultiPrecision.setPrecision(network, getPrecision());
      return (PipelineNetwork) network.freeze();
    }

  }

  public int getPixels(Tensor[] images) {
    return Math.max(1, Arrays.stream(images).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum());
  }

  @NotNull
  public MomentMatcher.MomentParams getMomentParams(PipelineNetwork network, double maskFactor, Tensor... images) {
    int pixels = getPixels(images);
    DAGNode mainIn = network.getHead();
    InnerNode avgNode = network.wrap(new BandAvgReducerLayer().setPrecision(getPrecision()), mainIn.addRef()); // Scale the average metrics by 1/x
    Tensor avgValue = eval(pixels, network, getTileSize(), 1.0, images);

    InnerNode recentered = network.wrap(new ImgBandDynamicBiasLayer().setPrecision(getPrecision()), mainIn,
        network.wrap(new ScaleLayer(-1 / maskFactor).setPrecision(getPrecision()), avgNode.addRef()));
    InnerNode rmsNode = network.wrap(new NthPowerActivationLayer().setPower(0.5),
        network.wrap(new ScaleLayer(1 / maskFactor).setPrecision(getPrecision()),
            network.wrap(new BandAvgReducerLayer().setPrecision(getPrecision()), // Scale the avg sq metrics by 1/x
                network.wrap(new SquareActivationLayer().setPrecision(getPrecision()), recentered.addRef())
            )
        )
    );
    Tensor rmsValue = eval(pixels, network, getTileSize(), 2.0, images);

    InnerNode rescaled = network.wrap(new ProductLayer().setPrecision(getPrecision()), recentered,
        network.wrap(new BoundedActivationLayer().setMinValue(0.0).setMaxValue(1e4),
            network.wrap(new NthPowerActivationLayer().setPower(-1), rmsNode.addRef())));


    InnerNode covNode = network.wrap(new ScaleLayer(1 / maskFactor).setPrecision(getPrecision()),
        network.wrap(new GramianLayer(getAppendUUID(network, GramianLayer.class)).setPrecision(getPrecision()), rescaled)
    ); // Scale the gram matrix by 1/x (elements are averages)
    Tensor covValue = eval(pixels, network, getTileSize(), 1.0, images);

    return new MomentParams(this, avgNode, avgValue, rmsNode, rmsValue, covNode, covValue);
  }

  @NotNull
  public MomentMatcher.MomentParams getMomentNodes(PipelineNetwork network, double maskFactor) {
    DAGNode mainIn = network.getHead();
    InnerNode avgNode = network.wrap(new BandAvgReducerLayer().setPrecision(getPrecision()), mainIn.addRef()); // Scale the average metrics by 1/x

    InnerNode recentered = network.wrap(new ImgBandDynamicBiasLayer().setPrecision(getPrecision()), mainIn,
        network.wrap(new ScaleLayer(-1 / maskFactor).setPrecision(getPrecision()), avgNode.addRef()));
    InnerNode rmsNode = network.wrap(new NthPowerActivationLayer().setPower(0.5),
        network.wrap(new ScaleLayer(1 / maskFactor).setPrecision(getPrecision()),
            network.wrap(new BandAvgReducerLayer().setPrecision(getPrecision()), // Scale the avg sq metrics by 1/x
                network.wrap(new SquareActivationLayer().setPrecision(getPrecision()), recentered.addRef())
            )
        )
    );

    InnerNode rescaled = network.wrap(new ProductLayer().setPrecision(getPrecision()), recentered,
        network.wrap(new BoundedActivationLayer().setMinValue(0.0).setMaxValue(1e4),
            network.wrap(new NthPowerActivationLayer().setPower(-1), rmsNode.addRef())));


    InnerNode covNode = network.wrap(new ScaleLayer(1 / maskFactor).setPrecision(getPrecision()),
        network.wrap(new GramianLayer(getAppendUUID(network, GramianLayer.class)).setPrecision(getPrecision()), rescaled)
    ); // Scale the gram matrix by 1/x (elements are averages)

    return new MomentParams(this, avgNode, null, rmsNode, null, covNode, null);
  }

  protected Tensor avg(PipelineNetwork network, int pixels, Tensor[] image) {
//    return eval(pixels, network, getTileSize(), 1.0, image);
    PipelineNetwork avgNet = PipelineNetwork.wrap(1,
        network.addRef(),
        new BandAvgReducerLayer()
    );
    Tensor evalRoot = eval(pixels, avgNet, getTileSize(), 1.0, image);
    avgNet.freeRef();
    return evalRoot;
  }


  public int getTileSize() {
    return tileSize;
  }

  public MomentMatcher setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  public double getPosCoeff() {
    return posCoeff;
  }

  public MomentMatcher setPosCoeff(double posCoeff) {
    this.posCoeff = posCoeff;
    return this;
  }

  public double getScaleCoeff() {
    return scaleCoeff;
  }

  public MomentMatcher setScaleCoeff(double scaleCoeff) {
    this.scaleCoeff = scaleCoeff;
    return this;
  }

  public double getCovCoeff() {
    return covCoeff;
  }

  public MomentMatcher setCovCoeff(double covCoeff) {
    this.covCoeff = covCoeff;
    return this;
  }

  public Precision getPrecision() {
    return precision;
  }

  public MomentMatcher setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }

  private static class MomentParams {
    private final InnerNode avgNode;
    private final Tensor avgValue;
    private final InnerNode rmsNode;
    private final Tensor rmsValue;
    private final InnerNode covNode;
    private final Tensor covValue;
    private final MomentMatcher momentMatcher;

    private MomentParams(MomentMatcher momentMatcher, InnerNode avgNode, Tensor avgValue, InnerNode rmsNode, Tensor rmsValue, InnerNode covNode, Tensor covValue) {
      this.momentMatcher = momentMatcher;
      this.avgNode = avgNode;
      this.avgValue = avgValue;
      this.rmsNode = rmsNode;
      this.rmsValue = rmsValue;
      this.covNode = covNode;
      this.covValue = covValue;
    }

    public InnerNode addLoss(PipelineNetwork network) {
      return network.wrap(new SumInputsLayer().setPrecision(momentMatcher.getPrecision()),
          network.wrap(new ScaleLayer(momentMatcher.getPosCoeff()).setPrecision(momentMatcher.getPrecision()),
              network.wrap(lossSq(momentMatcher.getPrecision(), getAvgValue()), getAvgNode())
          ),
          network.wrap(new ScaleLayer(momentMatcher.getScaleCoeff()).setPrecision(momentMatcher.getPrecision()),
              network.wrap(lossSq(momentMatcher.getPrecision(), getRmsValue()), getRmsNode())
          ),
          network.wrap(new ScaleLayer(momentMatcher.getCovCoeff()).setPrecision(momentMatcher.getPrecision()),
              network.wrap(lossSq(momentMatcher.getPrecision(), getCovValue()), getCovNode())
          )
      );
    }

    public InnerNode getAvgNode() {
      return avgNode;
    }

    public Tensor getAvgValue() {
      return avgValue;
    }

    public InnerNode getRmsNode() {
      return rmsNode;
    }

    public Tensor getRmsValue() {
      return rmsValue;
    }

    public InnerNode getCovNode() {
      return covNode;
    }

    public Tensor getCovValue() {
      return covValue;
    }
  }
}
