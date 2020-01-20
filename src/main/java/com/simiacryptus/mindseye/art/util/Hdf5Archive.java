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

package com.simiacryptus.mindseye.art.util;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.wrappers.*;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.hdf5;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.lang.Exception;
import java.nio.ByteBuffer;
import java.util.regex.Pattern;

import static org.bytedeco.javacpp.hdf5.*;

public class Hdf5Archive {
  private static final Logger log = LoggerFactory.getLogger(Hdf5Archive.class);

  static {
    try {
      /* This is necessary for the apply to the BytePointer constructor below. */
      Loader.load(hdf5.class);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  @Nonnull
  private final H5File file;
  @Nonnull
  private final File filename;

  public Hdf5Archive(@Nonnull String filename) {
    this(new File(filename));
  }

  public Hdf5Archive(@Nonnull File filename) {
    this.filename = filename;
    final String canonicalPath;
    try {
      canonicalPath = filename.getCanonicalPath();
    } catch (IOException e) {
      throw new RuntimeException(RefString.format("Error with filename %s", filename), e);
    }
    try {
      this.file = new H5File(canonicalPath, H5F_ACC_RDONLY());
    } catch (Exception e) {
      throw new RuntimeException(RefString.format("Error opening %s", canonicalPath), e);
    }
  }

  @Nonnull
  public File getFilename() {
    return filename;
  }

  private static void print(@Nonnull Hdf5Archive archive, @Nonnull Logger log) {
    printTree(archive, "", false, log);
  }

  private static void printTree(@Nonnull Hdf5Archive hdf5, CharSequence prefix, boolean printData, @Nonnull Logger log,
                                @Nonnull String... path) {
    for (CharSequence datasetName : hdf5.getDataSets(path)) {
      @Nullable
      Tensor tensor = hdf5.readDataSet(datasetName.toString(), path);
      assert tensor != null;
      log.info(RefString.format("%sDataset %s: %s", prefix, datasetName, RefArrays.toString(tensor.getDimensions())));
      if (printData)
        log.info(RefString.format("%s%s", prefix, tensor.prettyPrint().replaceAll("\n", "\n" + prefix)));
      tensor.freeRef();
    }
    hdf5.getAttributes(path).forEach((k, v) -> {
      log.info((RefString.format("%sAttribute: %s => %s", prefix, k, v)));
    });
    for (String t : hdf5.getGroups(path).stream().map(CharSequence::toString).sorted(new RefComparator<String>() {
      @Override
      public int compare(@Nonnull String o1, @Nonnull String o2) {
        @Nonnull
        String prefix = "layer_";
        @Nonnull
        Pattern digit = Pattern.compile("^\\d+$");
        if (digit.matcher(o1).matches() && digit.matcher(o2).matches())
          return Integer.compare(Integer.parseInt(o1), Integer.parseInt(o2));
        if (o1.startsWith(prefix) && o2.startsWith(prefix))
          return compare(o1.substring(prefix.length()), o2.substring(prefix.length()));
        else
          return o1.compareTo(o2);
      }
    }).collect(RefCollectors.toList())) {
      log.info(prefix + t);
      printTree(hdf5, prefix + "\t", printData, log, concat(path, t));
    }
  }

  @Nonnull
  private static String[] concat(@Nonnull CharSequence[] s, String t) {
    @Nonnull
    String[] strings = new String[s.length + 1];
    RefSystem.arraycopy(s, 0, strings, 0, s.length);
    strings[s.length] = t;
    return strings;
  }

  @Nonnull
  @Override
  public String toString() {
    return RefString.format("Hdf5Archive{%s}", file);
  }

  @Nullable
  public Tensor readDataSet(@Nonnull CharSequence datasetName, @Nonnull CharSequence... groups) {
    if (groups.length == 0) {
      return readDataSet(this.file, datasetName);
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    @Nullable
    Tensor a = readDataSet(groupArray[groupArray.length - 1], datasetName);
    closeGroups(groupArray);
    return a;
  }

  @Nullable
  public CharSequence readAttributeAsJson(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return readAttributeAsJson(this.file.openAttribute(attributeName));
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    @Nullable
    String s = readAttributeAsJson(groupArray[groups.length - 1].openAttribute(attributeName));
    closeGroups(groupArray);
    return s;
  }

  @Nullable
  public CharSequence readAttributeAsString(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return readAttributeAsString(this.file.openAttribute(attributeName));
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    @Nullable
    String s = readAttributeAsString(groupArray[groupArray.length - 1].openAttribute(attributeName));
    closeGroups(groupArray);
    return s;
  }

  public boolean hasAttribute(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return this.file.attrExists(attributeName);
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    boolean b = groupArray[groupArray.length - 1].attrExists(attributeName);
    closeGroups(groupArray);
    return b;
  }

  @Nonnull
  public RefMap<CharSequence, Object> getAttributes(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getAttributes(this.file);
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    Group group = groupArray[groupArray.length - 1];
    @Nonnull
    RefMap<CharSequence, Object> attributes = getAttributes(group);
    closeGroups(groupArray);
    return attributes;
  }

  @Nonnull
  public RefMap<CharSequence, Object> getAttributes(@Nonnull Group group) {
    int numAttrs = group.getNumAttrs();
    @Nonnull
    RefTreeMap<CharSequence, Object> attributes = new RefTreeMap<>();
    for (int i = 0; i < numAttrs; i++) {
      Attribute attribute = group.openAttribute(i);
      CharSequence name = attribute.getName().getString();
      int typeId = attribute.getTypeClass();
      if (typeId == 0) {
        attributes.put(name, getI64(attribute));
      } else {
        RefSystem.out.println(name + " type = " + typeId);
        attributes.put(name, getString(attribute));
      }
      attribute.deallocate();
    }
    return attributes;
  }

  @Nonnull
  public RefList<CharSequence> getDataSets(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getObjects(this.file, H5O_TYPE_DATASET);
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    @Nonnull
    RefList<CharSequence> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_DATASET);
    closeGroups(groupArray);
    return ls;
  }

  @Nonnull
  public RefList<CharSequence> getGroups(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getObjects(this.file, H5O_TYPE_GROUP);
    }
    @Nonnull
    Group[] groupArray = openGroups(groups);
    @Nonnull
    RefList<CharSequence> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_GROUP);
    closeGroups(groupArray);
    return ls;
  }

  @Nonnull
  public CharSequence readAttributeAsFixedLengthString(String attributeName, int bufferSize) {
    return readAttributeAsFixedLengthString(this.file.openAttribute(attributeName), bufferSize);
  }

  public void print() {
    print(log);
  }

  public void print(@Nonnull Logger log) {
    print(this, log);
  }

  @Nonnull
  private Group[] openGroups(@Nonnull CharSequence... groups) {
    @Nonnull
    Group[] groupArray = new Group[groups.length];
    groupArray[0] = this.file.openGroup(groups[0].toString());
    for (int i = 1; i < groups.length; i++) {
      groupArray[i] = groupArray[i - 1].openGroup(groups[i].toString());
    }
    return groupArray;
  }

  private void closeGroups(@Nonnull Group[] groupArray) {
    for (int i = groupArray.length - 1; i >= 0; i--) {
      groupArray[i].deallocate();
    }
  }

  private long getI64(@Nonnull Attribute attribute) {
    return getI64(attribute, attribute.getIntType(), new byte[8]);
  }

  @Nonnull
  private CharSequence getString(@Nonnull Attribute attribute) {
    return getString(attribute, attribute.getVarLenType(), new byte[1024]);
  }

  private long getI64(@Nonnull Attribute attribute, DataType dataType, @Nonnull byte[] buffer) {
    @Nonnull
    BytePointer pointer = new BytePointer(buffer);
    attribute.read(dataType, pointer);
    pointer.get(buffer);
    ArrayUtils.reverse(buffer);
    return ByteBuffer.wrap(buffer).asLongBuffer().get();
  }

  @Nonnull
  private CharSequence getString(@Nonnull Attribute attribute, DataType dataType, @Nonnull byte[] buffer) {
    @Nonnull
    BytePointer pointer = new BytePointer(buffer);
    attribute.read(dataType, pointer);
    pointer.get(buffer);
    @Nonnull
    String str = new String(buffer);
    if (str.indexOf('\0') >= 0) {
      return str.substring(0, str.indexOf('\0'));
    } else {
      return str;
    }
  }

  @Nonnull
  private Tensor readDataSet(@Nonnull Group fileGroup, @Nonnull CharSequence datasetName) {
    DataSet dataset = fileGroup.openDataSet(datasetName.toString());
    DataSpace space = dataset.getSpace();
    try {
      int nbDims = space.getSimpleExtentNdims();
      @Nonnull
      long[] dims = new long[nbDims];
      space.getSimpleExtentDims(dims);
      @Nullable
      float[] dataBuffer = null;
      @Nullable
      FloatPointer fp = null;
      int j = 0;
      @Nonnull
      DataType dataType = new DataType(PredType.NATIVE_FLOAT());
      @Nullable
      Tensor data = null;
      switch (nbDims) {
        case 4: /* 2D Convolution weights */
          dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2] * dims[3])];
          fp = new FloatPointer(dataBuffer);
          dataset.read(fp, dataType);
          fp.get(dataBuffer);
          data = new Tensor((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
          for (int i1 = 0; i1 < dims[0]; i1++)
            for (int i2 = 0; i2 < dims[1]; i2++)
              for (int i3 = 0; i3 < dims[2]; i3++)
                for (int i4 = 0; i4 < dims[3]; i4++)
                  data.set(i1, i2, i3, i4, dataBuffer[j++]);
          break;
        case 3:
          dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2])];
          fp = new FloatPointer(dataBuffer);
          dataset.read(fp, dataType);
          fp.get(dataBuffer);
          data = new Tensor((int) dims[0], (int) dims[1], (int) dims[2]);
          for (int i1 = 0; i1 < dims[0]; i1++)
            for (int i2 = 0; i2 < dims[1]; i2++)
              for (int i3 = 0; i3 < dims[2]; i3++)
                data.set(i1, i2, i3, dataBuffer[j++]);
          break;
        case 2: /* Dense and Recurrent weights */
          dataBuffer = new float[(int) (dims[0] * dims[1])];
          fp = new FloatPointer(dataBuffer);
          dataset.read(fp, dataType);
          fp.get(dataBuffer);
          data = new Tensor((int) dims[0], (int) dims[1]);
          for (int i1 = 0; i1 < dims[0]; i1++)
            for (int i2 = 0; i2 < dims[1]; i2++)
              data.set(i1, i2, dataBuffer[j++]);
          break;
        case 1: /* Bias */
          dataBuffer = new float[(int) dims[0]];
          fp = new FloatPointer(dataBuffer);
          dataset.read(fp, dataType);
          fp.get(dataBuffer);
          data = new Tensor((int) dims[0]);
          for (int i1 = 0; i1 < dims[0]; i1++) {
            final double value = dataBuffer[j++];
            data.set(i1, value);
          }
          break;
        default:
          throw new RuntimeException("Cannot import weights apply rank " + nbDims);
      }
      return data;
    } finally {
      space.deallocate();
      dataset.deallocate();
      dataset.close();
    }
  }

  @Nonnull
  private RefList<CharSequence> getObjects(@Nonnull Group fileGroup, int objType) {
    @Nonnull
    RefList<CharSequence> groups = new RefArrayList<CharSequence>();
    for (int i = 0; i < fileGroup.getNumObjs(); i++) {
      BytePointer objPtr = fileGroup.getObjnameByIdx(i);
      if (fileGroup.childObjType(objPtr) == objType) {
        groups.add(fileGroup.getObjnameByIdx(i).getString());
      }
    }
    return groups;
  }

  @Nonnull
  private String readAttributeAsJson(@Nonnull Attribute attribute) {
    VarLenType vl = attribute.getVarLenType();
    int bufferSizeMult = 1;
    @Nullable
    String s = null;
    /* TODO: find a less hacky way to do this.
     * Reading variable length strings (from attributes) is a giant
     * pain. There does not appear to be any way to determine the
     * length of the string in advance, so we use a hack: choose a
     * buffer size and read the config. If Jackson fails to parse
     * it, then we must not have read the entire config. Increase
     * buffer and repeat.
     */
    while (true) {
      @Nonnull
      byte[] attrBuffer = new byte[bufferSizeMult * 2000];
      @Nonnull
      BytePointer attrPointer = new BytePointer(attrBuffer);
      attribute.read(vl, attrPointer);
      attrPointer.get(attrBuffer);
      s = new String(attrBuffer);
      @Nonnull
      ObjectMapper mapper = new ObjectMapper();
      mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
      try {
        mapper.readTree(s);
        break;
      } catch (IOException e) {
      }
      bufferSizeMult++;
      if (bufferSizeMult > 100) {
        throw new RuntimeException("Could not read abnormally long HDF5 attribute");
      }
    }
    return s;
  }

  @Nonnull
  private String readAttributeAsString(@Nonnull Attribute attribute) {
    VarLenType vl = attribute.getVarLenType();
    int bufferSizeMult = 1;
    @Nullable
    String s = null;
    /* TODO: find a less hacky way to do this.
     * Reading variable length strings (from attributes) is a giant
     * pain. There does not appear to be any way to determine the
     * length of the string in advance, so we use a hack: choose a
     * buffer size and read the config, increase buffer and repeat
     * until the buffer ends apply \u0000
     */
    while (true) {
      @Nonnull
      byte[] attrBuffer = new byte[bufferSizeMult * 2000];
      @Nonnull
      BytePointer attrPointer = new BytePointer(attrBuffer);
      attribute.read(vl, attrPointer);
      attrPointer.get(attrBuffer);
      s = new String(attrBuffer);

      if (s.endsWith("\u0000")) {
        s = s.replace("\u0000", "");
        break;
      }

      bufferSizeMult++;
      if (bufferSizeMult > 100) {
        throw new RuntimeException("Could not read abnormally long HDF5 attribute");
      }
    }

    return s;
  }

  @Nonnull
  private CharSequence readAttributeAsFixedLengthString(@Nonnull Attribute attribute, int bufferSize) {
    VarLenType vl = attribute.getVarLenType();
    @Nonnull
    byte[] attrBuffer = new byte[bufferSize];
    @Nonnull
    BytePointer attrPointer = new BytePointer(attrBuffer);
    attribute.read(vl, attrPointer);
    attrPointer.get(attrBuffer);
    @Nonnull
    String s = new String(attrBuffer);
    return s;
  }
}
