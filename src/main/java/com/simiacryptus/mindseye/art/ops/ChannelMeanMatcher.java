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
import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.art.VisualModifierParameters;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * The type Channel mean matcher.
 */
public class ChannelMeanMatcher implements VisualModifier {

  private boolean balanced = true;
  private boolean averaging = true;
  private int tileSize = ArtSettings.INSTANCE().defaultTileSize;

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
   * @return the averaging
   */
  @Nonnull
  public ChannelMeanMatcher setAveraging(boolean averaging) {
    this.averaging = averaging;
    return this;
  }

  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return balanced;
  }

  /**
   * Sets balanced.
   *
   * @param balanced the balanced
   * @return the balanced
   */
  @Nonnull
  public ChannelMeanMatcher setBalanced(boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  @Nonnull
  @Override
  public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
    final PipelineNetwork pipelineNetwork = buildWithModel(visualModifierParameters.getNetwork(), null,
        visualModifierParameters.getStyle());
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(visualModifierParameters.scale);
    linearActivationLayer.freeze();
    pipelineNetwork.add(linearActivationLayer).freeRef();
    visualModifierParameters.freeRef();
    return pipelineNetwork;
  }

  /**
   * Build with model pipeline network.
   *
   * @param network    the network
   * @param meanSignal the mean signal
   * @param image      the image
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork buildWithModel(PipelineNetwork network, @Nullable Tensor meanSignal, @Nonnull Tensor... image) {
    PipelineNetwork copyPipeline = network.copyPipeline();
    network.freeRef();
    assert copyPipeline != null;
    RefUtil.freeRef(copyPipeline.add(new BandAvgReducerLayer()));
    if (meanSignal == null) {
      final PipelineNetwork meanNetwork = PipelineNetwork.build(1,
          new com.simiacryptus.mindseye.layers.java.ImgTileSubnetLayer(copyPipeline.addRef(), tileSize, tileSize), new BandAvgReducerLayer());
      Tensor tensor1 = RefUtil.get(RefArrays.stream(RefUtil.addRef(image)).map(tensor -> Result.getData0(meanNetwork.eval(tensor))).reduce((a, b) -> {
        return Tensor.add(a, b);
      }));
      tensor1.scaleInPlace(1.0 / image.length);
      RefUtil.freeRef(meanSignal);
      meanSignal = tensor1;
      meanNetwork.freeRef();
    }
    RefUtil.freeRef(image);
    double mag = isBalanced() ? meanSignal.rms() : 1;
    meanSignal.scaleInPlace(-1);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    double scale;
    if (Double.isFinite(mag) && mag > 0) {
      scale = Math.pow(mag, -2);
    } else {
      scale = 1;
    }
    linearActivationLayer.setScale(scale);
    Layer layer = PipelineNetwork.build(1,
        new ImgBandBiasLayer(meanSignal),
        new SquareActivationLayer(),
        isAveraging() ? new AvgReducerLayer() : new SumReducerLayer(),
        linearActivationLayer
    );
    layer.setName(RefString.format("RMS[x-C] / %.0E", mag));
    copyPipeline.add(layer).freeRef();
    copyPipeline.freeze();
    return copyPipeline;
  }
}
