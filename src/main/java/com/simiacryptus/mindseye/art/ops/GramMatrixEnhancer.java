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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumReducerLayer;
import com.simiacryptus.mindseye.layers.java.AbsActivationLayer;
import com.simiacryptus.mindseye.layers.java.BoundedActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class GramMatrixEnhancer implements VisualModifier {
  private static final Logger log = LoggerFactory.getLogger(GramMatrixEnhancer.class);
  private final Precision precision = Precision.Float;
  private double min = -1;
  private double max = 1;
  private boolean averaging = true;
  private boolean balanced = true;
  private int tileSize = 600;
  private int padding = 8;

  @NotNull
  public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
    PipelineNetwork rmsNetwork = new PipelineNetwork(1);
    rmsNetwork.setName(String.format("-RMS[x*C] / %.0E", mag));
    rmsNetwork.wrap(averaging ? new AvgReducerLayer() : new SumReducerLayer(),
        rmsNetwork.wrap(new BoundedActivationLayer().setMinValue(getMin()).setMaxValue(getMax()),
            rmsNetwork.wrap(new LinearActivationLayer().setScale(-Math.pow(mag, -2)),
                rmsNetwork.wrap(new ProductLayer(), rmsNetwork.getInput(0), rmsNetwork.constValueWrap(result))
            )
        )).freeRef();
    return rmsNetwork;
  }

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor... image) {
    network = MultiPrecision.setPrecision(network.copyPipeline(), precision);
    network.wrap(new GramianLayer(GramMatrixMatcher.getAppendUUID(network, GramianLayer.class)).setPrecision(precision)).freeRef();
    int pixels = Arrays.stream(image).mapToInt(x -> {
      int[] dimensions = x.getDimensions();
      return dimensions[0] * dimensions[1];
    }).sum();
    Tensor result = GramMatrixMatcher.eval(pixels, network, getTileSize(), padding, image);
    double mag = balanced ? result.rms() : 1;
    network.wrap(loss(result, mag, isAveraging())).freeRef();
    return (PipelineNetwork) network.freeze();
  }

  public boolean isAveraging() {
    return averaging;
  }

  public GramMatrixEnhancer setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public GramMatrixEnhancer setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  public int getTileSize() {
    return tileSize;
  }

  public GramMatrixEnhancer setTileSize(int tileSize) {
    this.tileSize = tileSize;
    return this;
  }

  public double getMax() {
    return max;
  }

  public GramMatrixEnhancer setMax(double max) {
    this.max = max;
    return this;
  }

  public double getMin() {
    return min;
  }

  public GramMatrixEnhancer setMin(double min) {
    this.min = min;
    return this;
  }

  public GramMatrixEnhancer setMinMax(double minValue, double maxValue) {
    this.min = minValue;
    this.max = maxValue;
    return this;
  }

  public static class StaticGramMatrixEnhancer extends GramMatrixEnhancer {
    @NotNull
    public PipelineNetwork loss(Tensor result, double mag, boolean averaging) {
      PipelineNetwork rmsNetwork = new PipelineNetwork(1);
      rmsNetwork.setName(String.format("-RMS[x*C] / %.0E", mag));
      rmsNetwork.wrap(averaging ? new AvgReducerLayer() : new SumReducerLayer(),
          rmsNetwork.wrap(new BoundedActivationLayer().setMinValue(getMin()).setMaxValue(getMax()),
              rmsNetwork.wrap(new LinearActivationLayer().setScale(-Math.pow(mag, -2)),
                  rmsNetwork.wrap(new AbsActivationLayer())
              )
          )).freeRef();
      return rmsNetwork;
    }
  }
}
