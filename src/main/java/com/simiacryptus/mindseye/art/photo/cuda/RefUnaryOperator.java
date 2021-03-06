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

import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import javax.annotation.Nonnull;
import java.util.function.UnaryOperator;

/**
 * The interface Ref unary operator.
 *
 * @param <T> the type parameter
 */
public interface RefUnaryOperator<T> extends ReferenceCounting, UnaryOperator<T> {
  /**
   * Wrap ref unary operator.
   *
   * @param <T>   the type parameter
   * @param inner the inner
   * @return the ref unary operator
   */
  @Nonnull
  static <T> RefUnaryOperator<T> wrap(UnaryOperator<T> inner) {
    return new RefUnaryOperatorWrapper<T>(inner);
  }


  /**
   * Iterate t.
   *
   * @param n   the n
   * @param obj the obj
   * @return the t
   */
  default T iterate(int n, T obj) {
    return n <= 1 ? apply(obj) : this.iterate(n - 1, apply(obj));
  }

  /**
   * Free.
   */
  void _free();

  @Nonnull
  RefUnaryOperator<T> addRef();

  /**
   * The type Ref unary operator wrapper.
   *
   * @param <T> the type parameter
   */
  class RefUnaryOperatorWrapper<T> extends ReferenceCountingBase implements RefUnaryOperator<T> {

    private UnaryOperator<T> inner;

    /**
     * Instantiates a new Ref unary operator wrapper.
     *
     * @param inner the inner
     */
    public RefUnaryOperatorWrapper(UnaryOperator<T> inner) {
      this.inner = inner;
    }

    @Override
    public T apply(T t) {
      return inner.apply(t);
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    RefUnaryOperatorWrapper<T> addRef() {
      return (RefUnaryOperatorWrapper<T>) super.addRef();
    }
  }
}
