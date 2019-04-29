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
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.UUID;

public class GramMatrixMatcher implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixMatcher.class);
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = 600;

  @NotNull
  public static Layer loss(Tensor result, double mag, boolean averaging) {
    Layer layer = PipelineNetwork.wrap(1,
        new ImgBandBiasLayer(result.scaleInPlace(-1)),
        new SquareActivationLayer(),
        averaging ? new AvgReducerLayer() : new SumReducerLayer(),
        new NthPowerActivationLayer().setPower(0.5),
        new LinearActivationLayer().setScale(Math.pow(mag, -1))
    ).setName(String.format("RMS[x-C] / %.0E", mag));
    result.freeRef();
    return layer;
  }

  public static Tensor eval(int pixels, PipelineNetwork network, int tileSize, Tensor... image) {
    return Arrays.stream(image).flatMap(img -> {
      int[] imageDimensions = img.getDimensions();
      return Arrays.stream(TiledTrainable.selectors(0, imageDimensions[0], imageDimensions[1], tileSize, CudaSettings.INSTANCE().defaultPrecision))
          .map(s -> s.getCompatibilityLayer())
          .map(selector -> {
            //log.info(selector.toString());
            Tensor tile = selector.eval(img).getDataAndFree().getAndFree(0);
            selector.freeRef();
            int[] tileDimensions = tile.getDimensions();
            Tensor component = network.eval(tile).getDataAndFree().getAndFree(0).scaleInPlace(tileDimensions[0] * tileDimensions[1]);
            tile.freeRef();
            return component;
          });
    }).reduce((a, b) -> {
      a.addInPlace(b);
      b.freeRef();
      return a;
    }).get().scaleInPlace(1.0 / pixels).mapAndFree(x -> {
      if (Double.isFinite(x)) {
        return x;
      } else {
        return 0;
      }
    });
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
    network = network.copyPipeline();
    network.wrap(new GramianLayer(getAppendUUID(network, GramianLayer.class))).freeRef();
    int pixels = Arrays.stream(image).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum();
    Tensor result = eval(pixels, network, getTileSize(), image);
    double mag = balanced ? result.rms() : 1;
    network.wrap(loss(result, mag, isAveraging())).freeRef();
    return (PipelineNetwork) network.freeze();
  }

  public boolean isAveraging() {
    return averaging;
  }

  public GramMatrixMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public GramMatrixMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  public int getTileSize() {
    return tileSize;
  }

  public GramMatrixMatcher setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }
}