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

package com.simiacryptus.mindseye.art.photo.cuda;

import com.simiacryptus.ref.lang.ReferenceCountingBase;

import java.util.Arrays;

class SingleChannelWrapper extends ReferenceCountingBase implements RefOperator<double[][]> {
  private final RefOperator<double[][]> unaryOperator;

  public SingleChannelWrapper(RefOperator<double[][]> unaryOperator) {
    this.unaryOperator = unaryOperator;
  }

  @Override
  public double[][] apply(double[][] img) {
    return Arrays.stream(img)
        .map(x -> unaryOperator.apply(new double[][]{x})[0])
        .toArray(i -> new double[i][]);
  }

  @Override
  protected void _free() {
    this.unaryOperator.freeRef();
    super._free();
  }
}
