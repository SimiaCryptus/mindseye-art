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
  private final Precision precision = Precision.Float;
  private int tileSize = 600;
  private double posCoeff = 1.0;
  private double scaleCoeff = 1.0;
  private double covCoeff = 1.0;

  @NotNull
  public static Layer lossSq(Precision precision, Tensor target) {
    Layer layer = PipelineNetwork.wrap(1,
        new LinearActivationLayer().setScale(Math.pow(target.rms(), -1)),
        new ImgBandBiasLayer(target.scaleInPlace(-Math.pow(target.rms(), -1))).setPrecision(precision),
        new SquareActivationLayer().setPrecision(precision),
        new AvgReducerLayer().setPrecision(precision)
    ).setName(String.format("RMS[x-C] / %.0E", target.rms()));
    target.freeRef();
    return layer;
  }

  protected static Tensor eval(int pixels, PipelineNetwork network, int tileSize, Precision precision, double power, Tensor[] image) {
    return sum(Arrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      return Arrays.stream(TiledTrainable.selectors(0, imageDimensions[0], imageDimensions[1], tileSize, precision))
          .map(s -> s.getCompatibilityLayer())
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
      a.addInPlace(b);
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
  public PipelineNetwork build(PipelineNetwork network, Tensor... image) {
    network = (PipelineNetwork) MultiPrecision.setPrecision(network.copyPipeline(), precision);
    int pixels = Math.max(1, Arrays.stream(image).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum());
    DAGNode mainIn = network.getHead();

    InnerNode avg = network.wrap(new com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer().setPrecision(precision), mainIn.addRef());
    Tensor evalAvg = eval(pixels, network, getTileSize(), precision, 1.0, image);
    InnerNode recentered = network.wrap(new ImgBandDynamicBiasLayer().setPrecision(precision), mainIn,
        network.wrap(new ScaleLayer(-1).setPrecision(precision), avg.addRef()));
    ;

    InnerNode rms = network.wrap(new NthPowerActivationLayer().setPower(0.5),
        network.wrap(new com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer().setPrecision(precision),
            network.wrap(new SquareActivationLayer().setPrecision(precision), recentered.addRef())
        )
    );
    Tensor evalRms = eval(pixels, network, getTileSize(), precision, 1.0, image);
    InnerNode rescaled = network.wrap(new ProductLayer().setPrecision(precision), recentered,
        network.wrap(new BoundedActivationLayer().setMinValue(0.0).setMaxValue(1e4),
            network.wrap(new NthPowerActivationLayer().setPower(-1), rms.addRef())));

    InnerNode cov = network.wrap(new GramianLayer(getAppendUUID(network, GramianLayer.class)).setPrecision(precision), rescaled);
    Tensor evalCov = eval(pixels, network, getTileSize(), precision, 1.0, image);


    network.wrap(new SumInputsLayer().setPrecision(precision),
        network.wrap(new ScaleLayer(getPosCoeff()).setPrecision(precision),
            network.wrap(lossSq(precision, evalAvg), avg)
        ),
        network.wrap(new ScaleLayer(getScaleCoeff()).setPrecision(precision),
            network.wrap(lossSq(precision, evalRms), rms)
        ),
        network.wrap(new ScaleLayer(getCovCoeff()).setPrecision(precision),
            network.wrap(lossSq(precision, evalCov), cov)
        )
    );
    MultiPrecision.setPrecision(network, precision);
    return (PipelineNetwork) network.freeze();
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
}
