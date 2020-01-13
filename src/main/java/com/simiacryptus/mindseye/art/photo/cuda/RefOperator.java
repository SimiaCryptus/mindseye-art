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

import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;

import java.util.Arrays;
import java.util.function.UnaryOperator;

public interface RefOperator<T> extends ReferenceCounting, UnaryOperator<T> {
  static <T> RefOperator<T> wrap(UnaryOperator<T> inner) {
    return new RefOperatorWrapper<T>(inner);
  }

  public static @SuppressWarnings("unused") RefOperator[] addRefs(RefOperator[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RefOperator::addRef).toArray((x) -> new RefOperator[x]);
  }

  public static @SuppressWarnings("unused") RefOperator[][] addRefs(RefOperator[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RefOperator::addRefs).toArray((x) -> new RefOperator[x][]);
  }

  default T iterate(int n, T obj) {
    return n <= 1 ? apply(obj) : this.iterate(n - 1, apply(obj));
  }

  public void _free();

  public RefOperator<T> addRef();

  class RefOperatorWrapper<T> extends ReferenceCountingBase implements RefOperator<T> {

    private UnaryOperator<T> inner;

    public RefOperatorWrapper(UnaryOperator<T> inner) {
      this.inner = inner;
    }

    public static @SuppressWarnings("unused") RefOperatorWrapper[] addRefs(RefOperatorWrapper[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(RefOperatorWrapper::addRef)
          .toArray((x) -> new RefOperatorWrapper[x]);
    }

    @Override
    public T apply(T t) {
      return inner.apply(t);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") RefOperatorWrapper<T> addRef() {
      return (RefOperatorWrapper<T>) super.addRef();
    }
  }
}
