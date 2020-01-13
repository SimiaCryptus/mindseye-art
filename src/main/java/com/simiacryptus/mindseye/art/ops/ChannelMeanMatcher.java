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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;
import org.jetbrains.annotations.NotNull;

public class ChannelMeanMatcher implements VisualModifier {

  private boolean balanced = true;
  private boolean averaging = true;
  private int tileSize = 600;

  public boolean isAveraging() {
    return averaging;
  }

  public ChannelMeanMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  public ChannelMeanMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Override
  public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
    Tensor meanSignal = null;
    final PipelineNetwork pipelineNetwork = buildWithModel(visualModifierParameters.network, meanSignal,
        visualModifierParameters.style);
    visualModifierParameters.freeRef();
    return pipelineNetwork;
  }

  @NotNull
  public PipelineNetwork buildWithModel(PipelineNetwork network, Tensor meanSignal, Tensor... image) {
    network = network.copyPipeline();
    network.add(new BandAvgReducerLayer());
    if (meanSignal == null) {
      final PipelineNetwork meanNetwork = PipelineNetwork.build(1,
          new ImgTileSubnetLayer(network.addRef(), tileSize, tileSize), new BandAvgReducerLayer());
      meanSignal = RefUtil.get(RefArrays.stream(image).map(tensor -> meanNetwork.eval(tensor).getData().get(0)).reduce((a, b) -> {
        Tensor c = a.addAndFree(b);
        b.freeRef();
        return c;
      })).scaleInPlace(1.0 / image.length);
      meanNetwork.freeRef();
    }
    double mag = isBalanced() ? meanSignal.rms() : 1;
    final Layer[] layers = new Layer[] { new ImgBandBiasLayer(meanSignal.scaleInPlace(-1)), new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        new LinearActivationLayer().setScale(Math.pow(mag, -2)) };
    network.add(PipelineNetwork.build(1, layers).setName(RefString.format("RMS[x-C] / %.0E", mag))).freeRef();
    return (PipelineNetwork) network.freeze();
  }
}
