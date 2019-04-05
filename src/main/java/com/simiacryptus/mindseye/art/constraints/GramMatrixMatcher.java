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

package com.simiacryptus.mindseye.art.constraints;

import com.simiacryptus.mindseye.art.VisualModifier;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public class GramMatrixMatcher implements VisualModifier {

  @Override
  public PipelineNetwork build(PipelineNetwork network, Tensor image) {
    network = network.copy();
    Tensor rmsResult = network.eval(image).getDataAndFree().getAndFree(0).scaleInPlace(-1);
    double rms = Math.sqrt(rmsResult.sumSq() / rmsResult.length());
    rmsResult.freeRef();
    network.wrap(new GramianLayer()).freeRef();
    Tensor result = network.eval(image).getDataAndFree().getAndFree(0).scaleInPlace(-1);
    network.wrap(PipelineNetwork.wrap(1,
        new ImgBandBiasLayer(result),
        new SquareActivationLayer(),
        new AvgReducerLayer(),
        new NthPowerActivationLayer().setPower(0.5),
        new LinearActivationLayer().setScale(Math.pow(rms, -1))
    ).setName(String.format("RMS[x-C] / %.0E", rms))).freeRef();
    result.freeRef();
    return (PipelineNetwork) network.freeze();
  }

}
