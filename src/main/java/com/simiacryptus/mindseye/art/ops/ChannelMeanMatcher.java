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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class ChannelMeanMatcher implements VisualModifier {

  private boolean balanced = true;
  private boolean averaging = true;
  private int tileSize = 600;

  public boolean isAveraging() {
    return averaging;
  }

  @Nonnull
  public ChannelMeanMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  public boolean isBalanced() {
    return balanced;
  }

  @Nonnull
  public ChannelMeanMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    Tensor meanSignal = null;
    final PipelineNetwork pipelineNetwork = buildWithModel(visualModifierParameters.network, null,
        visualModifierParameters.style);
    visualModifierParameters.freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  public PipelineNetwork buildWithModel(PipelineNetwork network, @Nullable Tensor meanSignal, @Nonnull Tensor... image) {
    network = network.copyPipeline();
    assert network != null;
    network.add(new BandAvgReducerLayer());
    if (meanSignal == null) {
      final PipelineNetwork meanNetwork = PipelineNetwork.build(1,
          new ImgTileSubnetLayer(network.addRef(), tileSize, tileSize), new BandAvgReducerLayer());
      Tensor tensor1 = RefUtil.get(RefArrays.stream(image).map(tensor -> meanNetwork.eval(tensor).getData().get(0)).reduce((a, b) -> {
        Tensor c = a.addAndFree(b);
        b.freeRef();
        return c;
      }));
      tensor1.scaleInPlace(1.0 / image.length);
      meanSignal = tensor1.addRef();
      meanNetwork.freeRef();
    }
    double mag = isBalanced() ? meanSignal.rms() : 1;
    meanSignal.scaleInPlace(-1);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(Math.pow(mag, -2));
    final Layer[] layers = new Layer[]{new ImgBandBiasLayer(meanSignal.addRef()), new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        linearActivationLayer.addRef()};
    Layer layer = PipelineNetwork.build(1, layers);
    layer.setName(RefString.format("RMS[x-C] / %.0E", mag));
    network.add(layer.addRef()).freeRef();
    network.freeze();
    return network.addRef();
  }
}
