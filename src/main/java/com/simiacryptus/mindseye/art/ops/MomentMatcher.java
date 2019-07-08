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
        new LinearActivationLayer().setScale(Math.pow(rms, -1)),
        new ImgBandBiasLayer(target.scaleInPlace(-Math.pow(rms, -1))).setPrecision(precision),
        new SquareActivationLayer().setPrecision(precision),
        new AvgReducerLayer().setPrecision(precision)
    ).setName(String.format("RMS[x-C] / %.0E", rms));
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
        .mapAndFree(x -> Math.pow(x, 1.0 / power))
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

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor... images) {
    network = MultiPrecision.setPrecision(network.copyPipeline(), getPrecision());
    int pixels = Math.max(1, Arrays.stream(images).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum());
    DAGNode mainIn = network.getHead();

    Tensor evalRoot = avg(network, pixels, images);
    double sumSq = evalRoot.sumSq();
    log.info(String.format("Adjust for %s : %s", network.getName(), sumSq));
    network.wrap(new ScaleLayer(1.0/ sumSq));
    evalRoot.freeRef();

    InnerNode avg = network.wrap(new com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer().setPrecision(getPrecision()), mainIn.addRef());
    Tensor evalAvg = eval(pixels, network, getTileSize(), 1.0, images);
    InnerNode recentered = network.wrap(new ImgBandDynamicBiasLayer().setPrecision(getPrecision()), mainIn,
        network.wrap(new ScaleLayer(-1).setPrecision(getPrecision()), avg.addRef()));

    InnerNode rms = network.wrap(new NthPowerActivationLayer().setPower(0.5),
        network.wrap(new com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer().setPrecision(getPrecision()),
            network.wrap(new SquareActivationLayer().setPrecision(getPrecision()), recentered.addRef())
        )
    );
    Tensor evalRms = eval(pixels, network, getTileSize(), 2.0, images);
    InnerNode rescaled = network.wrap(new ProductLayer().setPrecision(getPrecision()), recentered,
        network.wrap(new BoundedActivationLayer().setMinValue(0.0).setMaxValue(1e4),
            network.wrap(new NthPowerActivationLayer().setPower(-1), rms.addRef())));

    InnerNode cov = network.wrap(new GramianLayer(getAppendUUID(network, GramianLayer.class)).setPrecision(getPrecision()), rescaled);
    Tensor evalCov = eval(pixels, network, getTileSize(), 1.0, images);


    network.wrap(new SumInputsLayer().setPrecision(getPrecision()),
        network.wrap(new ScaleLayer(getPosCoeff()).setPrecision(getPrecision()),
            network.wrap(lossSq(getPrecision(), evalAvg), avg)
        ),
        network.wrap(new ScaleLayer(getScaleCoeff()).setPrecision(getPrecision()),
            network.wrap(lossSq(getPrecision(), evalRms), rms)
        ),
        network.wrap(new ScaleLayer(getCovCoeff()).setPrecision(getPrecision()),
            network.wrap(lossSq(getPrecision(), evalCov), cov)
        )
    );
    MultiPrecision.setPrecision(network, getPrecision());
    return (PipelineNetwork) network.freeze();
  }

  protected Tensor avg(PipelineNetwork network, int pixels, Tensor[] image) {
    PipelineNetwork avgNet = PipelineNetwork.wrap(1,
        network.addRef(),
        new com.simiacryptus.mindseye.layers.java.AvgReducerLayer()
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
}
