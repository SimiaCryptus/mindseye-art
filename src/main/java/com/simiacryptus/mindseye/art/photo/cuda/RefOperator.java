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

import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.lang.ref.ReferenceCountingBase;

import java.util.function.UnaryOperator;

public interface RefOperator<T> extends ReferenceCounting, UnaryOperator<T> {
  static <T> RefOperator<T> wrap(UnaryOperator<T> inner) {
    return new RefOperatorWrapper<T>(inner);
  }

  default T iterate(int n, T obj) {
    return n <= 1 ? apply(obj) : this.iterate(n - 1, apply(obj));
  }

  class RefOperatorWrapper<T> extends ReferenceCountingBase implements RefOperator<T> {

    private UnaryOperator<T> inner;

    public RefOperatorWrapper(UnaryOperator<T> inner) {
      this.inner = inner;
    }

    @Override
    public T apply(T t) {
      return inner.apply(t);
    }
  }
}