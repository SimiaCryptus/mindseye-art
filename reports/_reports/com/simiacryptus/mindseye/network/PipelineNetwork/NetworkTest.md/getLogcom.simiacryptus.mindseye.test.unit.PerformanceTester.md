1. [Performance](#p-0)

# <a id="p-0"></a>Performance
<a id="p-1"></a>Now we execute larger-scale runs to benchmark performance:

<a id="p-2"></a>Code from [PerformanceTester.java:190](https://github.com/SimiaCryptus/mindseye-test/tree/444256810c541076ac97c4437963f8489a719862/src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L190) executed in 12.29 seconds (0.000 gc): 
```java
    test(component, inputPrototype);
```
<a id="p-3"></a>Logging: 
```
    1 batch length, 5 trials
    Input Dimensions:
    	[500, 333, 3]
    Performance:
    	Evaluation performance: 2.309906s +- 3.878533s [0.198457s - 10.048652s]
    	Learning performance: 0.011142s +- 0.002416s [0.009796s - 0.015972s]
    
```

<a id="p-4"></a>Per-key Performance Metrics:

<a id="p-5"></a>Code from [TestUtil.java:233](https://github.com/SimiaCryptus/mindseye-test/tree/444256810c541076ac97c4437963f8489a719862/src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L233) executed in 0.01 seconds (0.000 gc): 
```java
    @Nonnull final Map<CharSequence, MonitoringWrapperLayer> metrics = new HashMap<>();
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        @Nullable final MonitoringWrapperLayer layer = node.getLayer();
        Layer inner = layer.getInner();
        String str = inner.toString();
        str += " class=" + inner.getClass().getName();
```
<a id="p-6"></a>Logging: 
```
    Performance: 
    	2.206612s +- 3.888722s (5) <- GramianLayer/dee73032-3665-4705-b1fb-edb73fc197a4 class=com.simiacryptus.mindseye.layers.cudnn.GramianLayer
    	2.033112s +- 3.935404s (5) <- GramianLayer/e168e7a4-7545-422d-9940-46c96144e0eb class=com.simiacryptus.mindseye.layers.cudnn.GramianLayer
    	0.217915s +- 0.273672s (5) <- PipelineNetwork/c2d8ac34-b392-48c1-9c26-f6ab5d7e4f58 class=com.simiacryptus.mindseye.network.PipelineNetwork
    	0.143420s +- 0.264606s (5) <- ImgBandBiasLayer/cac41051-8462-4d9e-8f62-f9fd1dc5fc98 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.039229s +- 0.012116s (5) <- SimpleConvolutionLayer/e5b4d52f-92a0-43d4-a6bf-123a4a626c43 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.038610s +- 0.012598s (5) <- SimpleConvolutionLayer/fc087c6c-44db-4ed5-991d-ede7cde055d0 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.030956s +- 0.012495s (5) <- SimpleConvolutionLayer/278f5fdf-a9e2-492f-9b77-013ac9c71a4c class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.026902s +- 0.011049s (5) <- ImgBandBiasLayer/32a3b52b-e6d1-4165-afa4-1a4acbf14e8a class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.026044s +- 0.000444s (2) <- PipelineNetwork/967abef9-9058-45d8-82a6-1eeaefff2b10 class=com.simiacryptus.mindseye.network.PipelineNetwork
    	0.018110s +- 0.006399s (5) <- PipelineNetwork/57e5812c-e8f2-40be-a6ab-6a24b3f5d8c5 class=com.simiacryptus.mindseye.network.PipelineNetwork
    	Back: 0.003097s +- 0.000970s (5)
    	0.013765s +- 0.005724s (5) <- SimpleConvolutionLayer/e4606e4f-1b32-4c96-808f-066d872e55b7 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.013192s +- 0.000856s (5) <- SimpleConvolutionLayer/15053ddc-e2e6-4787-b061-f8b0b7f521fa class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.013085s +- 0.005094s (5) <- PipelineNetwork/8b849158-8efe-4cc0-81f7-a443c541e032 class=com.simiacryptus.mindseye.network.PipelineNetwork
    	Back: 0.003578s +- 0.000357s (5)
    	0.012556s +- 0.006635s (5) <- SimpleConvolutionLayer/36addb27-1a1f-4374-bded-47d7afbbae56 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.011887s +- 0.000324s (2) <- PipelineNetwork/542de883-4c7c-41f2-92c8-0a51054b73ca class=com.simiacryptus.mindseye.network.PipelineNetwork
    	0.011736s +- 0.006687s (5) <- ImgBandBiasLayer/d4bf606a-7910-48cb-b696-4bd5c100158f class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.009650s +- 0.006211s (5) <- ImgBandBiasLayer/d1351380-502c-4048-b3a4-7ad96f9a2aee class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.009608s +- 0.001827s (5) <- PipelineNetwork/1b5cb1bc-a68b-4856-98bc-21a867e25938 class=com.simiacryptus.mindseye.network.PipelineNetwork
    	Back: 0.002904s +- 0.000529s (5)
    	0.007574s +- 0.005438s (5) <- PoolingLayer/d34d2b21-ced2-48b1-89b4-baa5dfff1940 class=com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
    	0.007257s +- 0.003364s (5) <- SimpleConvolutionLayer/74bb5d37-3d22-49a5-84dc-e7a4ed8d40f4 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.005905s +- 0.002671s (5) <- SumInputsLayer/6d31252d-7b72-47ee-91a2-e5563707e43d class=com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer
    	0.004798s +- 0.001825s (5) <- ImgBandBiasLayer/239972ad-1d95-498a-983a-538d4351477b class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.004693s +- 0.001306s (5) <- PoolingLayer/b427938b-1129-4e90-ba13-f48283fa3c36 class=com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
    	0.004484s +- 0.001667s (5) <- ImgBandBiasLayer/e2fd5eb0-1573-4e40-901d-bae883b44049 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.004481s +- 0.000474s (5) <- ImgBandBiasLayer/068f5dc2-dd34-4d52-b49e-a9840d7bd573 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.004206s +- 0.000818s (5) <- SimpleConvolutionLayer/327faa47-9d3d-4c5f-9b4e-bd2a1237e959 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.003778s +- 0.000226s (4) <- PoolingLayer/c04f6fc6-d0c9-4cdd-a170-2f6704169f26 class=com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
    	0.003610s +- 0.001836s (5) <- SumInputsLayer/f603b63c-1c73-4ed2-a103-e592b266e636 class=com.simiacryptus.mindseye.layers.java.SumInputsLayer
    	Back: 0.000397s +- 0.000085s (5)
    	0.002558s +- 0.000738s (5) <- NthPowerActivationLayer/55b1eb80-2c75-46c8-9a0b-e13e114eb9cd class=com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer
    	Back: 0.000906s +- 0.000017s (5)
    	0.002324s +- 0.002623s (5) <- AvgReducerLayer/aed92374-e28e-4f4e-86ae-e7b35906eb4f class=com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer
    	Back: 0.001588s +- 0.000370s (5)
    	0.002248s +- 0.000647s (5) <- AvgReducerLayer/eaafaa61-98d6-41da-9926-80ee7df4b14a class=com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer
    	Back: 0.001064s +- 0.000336s (5)
    	0.002224s +- 0.000240s (5) <- ImgBandBiasLayer/5f5ae3f9-04b1-46f0-8f4b-5286394f3151 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.002128s +- 0.000406s (5) <- NthPowerActivationLayer/8b9322fe-f803-4f80-baef-7b7ba7c58825 class=com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer
    	Back: 0.000907s +- 0.000209s (5)
    	0.002010s +- 0.000773s (5) <- SimpleConvolutionLayer/cccc0a7a-4e01-4fd1-b2df-8c70f916ab83 class=com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
    	0.001778s +- 0.000653s (5) <- ImgConcatLayer/9e82c11f-f78c-44dc-8b8b-f7929e889ec6 class=com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer
    	0.001743s +- 0.000624s (5) <- ActivationLayer/fab99517-119e-4a85-9c3c-16c7f69c8031 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.001634s +- 0.000932s (5) <- AvgReducerLayer/0a9d3858-1122-41c9-8493-8b7545cee738 class=com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer
    	Back: 0.001017s +- 0.000216s (5)
    	0.001466s +- 0.000525s (5) <- NthPowerActivationLayer/c8deed21-997a-4440-83ab-d17622801575 class=com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer
    	Back: 0.000796s +- 0.000016s (5)
    	0.001436s +- 0.000731s (5) <- ImgBandBiasLayer/a7420849-484c-45a4-a88f-f6100fcc5cd5 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.001307s +- 0.000622s (5) <- ImgBandBiasLayer/79ef3e04-53f8-4017-8faf-d7e4470ea8e9 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.001284s +- 0.000834s (5) <- LinearActivationLayer/592a751f-654b-4d83-8800-d6e7e2d889f6 class=com.simiacryptus.mindseye.layers.java.LinearActivationLayer
    	Back: 0.000754s +- 0.000014s (5)
    	0.001223s +- 0.000432s (5) <- ImgBandBiasLayer/580f753d-e5c3-4b1c-8429-32d70031a6a3 class=com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
    	0.001131s +- 0.000448s (5) <- LRNLayer/970614fe-3738-4a48-9b0d-032a0d734c89 class=com.simiacryptus.mindseye.layers.cudnn.LRNLayer
    	0.001058s +- 0.000654s (5) <- ActivationLayer/19ebf7dd-17d1-4471-8090-1fc3e72af2f9 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.001019s +- 0.000465s (5) <- ActivationLayer/ce04c635-8c85-4fba-a897-cd6e7ef7bafe class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000986s +- 0.000332s (5) <- ActivationLayer/0a46a294-4fa4-4fab-9f03-f274fb56b2b3 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000985s +- 0.000744s (5) <- SquareActivationLayer/b5954860-bb2e-4d55-bcad-5353fd8001ec class=com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer
    	Back: 0.000297s +- 0.000037s (5)
    	0.000970s +- 0.000324s (5) <- ValueLayer/e87af9a1-64e6-4581-92de-6e59b13bfd0e class=com.simiacryptus.mindseye.layers.ValueLayer
    	0.000903s +- 0.000463s (5) <- ActivationLayer/32667d38-8086-4500-b58a-0e46a7c00467 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000880s +- 0.000599s (5) <- ActivationLayer/a2c6b7ee-7c5b-4f0b-8ee2-0f621f868802 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000827s +- 0.000580s (5) <- ActivationLayer/277b40fc-e541-43e0-899d-35cd69fa0e81 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000819s +- 0.000305s (5) <- ActivationLayer/d056f230-3fd8-455f-ac58-d3d498dff763 class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000814s +- 0.000425s (5) <- LRNLayer/fd728980-2e0c-4ed9-ab59-f86bfeea429e class=com.simiacryptus.mindseye.layers.cudnn.LRNLayer
    	0.000711s +- 0.000110s (5) <- LinearActivationLayer/045f6df1-0699-4249-8993-f112a98a661d class=com.simiacryptus.mindseye.layers.java.LinearActivationLayer
    	Back: 0.000541s +- 0.000008s (5)
    	0.000675s +- 0.000122s (5) <- ActivationLayer/af871086-194f-4e28-b36a-7eb9acab45db class=com.simiacryptus.mindseye.layers.cudnn.ActivationLayer
    	0.000617s +- 0.000059s (5) <- SquareActivationLayer/06ffb456-9fcd-4405-bfaf-987568bc0e5e class=com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer
    	Back: 0.000374s +- 0.000152s (5)
    	0.000569s +- 0.000044s (5) <- SquareActivationLayer/97adc093-9502-4f46-9608-5d5e60640fe3 class=com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer
    	Back: 0.000391s +- 0.000225s (5)
    	0.000313s +- 0.000083s (5) <- LinearActivationLayer/4193c78a-9c73-4d47-856e-a602a702e533 class=com.simiacryptus.mindseye.layers.java.LinearActivationLayer
    	Back: 0.000685s +- 0.000149s (5)
    	0.000278s +- 0.000004s (5) <- LinearActivationLayer/492e283d-b40a-4306-bb07-eeaf7cd7f52d class=com.simiacryptus.mindseye.layers.java.LinearActivationLayer
    	Back: 0.000674s +- 0.000142s (5)
    
```
