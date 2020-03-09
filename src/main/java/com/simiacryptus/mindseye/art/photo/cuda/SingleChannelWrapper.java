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

import javax.annotation.Nonnull;
import java.util.Arrays;

class SingleChannelWrapper extends ReferenceCountingBase implements RefUnaryOperator<double[][]> {
  private final RefUnaryOperator<double[][]> unaryOperator;

  public SingleChannelWrapper(RefUnaryOperator<double[][]> unaryOperator) {
    this.unaryOperator = unaryOperator;
  }

  @Nonnull
  @Override
  public double[][] apply(@Nonnull double[][] img) {
    return Arrays.stream(img).map(x -> unaryOperator.apply(new double[][]{x})[0]).toArray(i -> new double[i][]);
  }

  @Override
  public SingleChannelWrapper addRef() {
    return (SingleChannelWrapper) super.addRef();
  }

  @Override
  public void _free() {
    super._free();
    unaryOperator.freeRef();
  }
}
