1. [Performance](#p-0)

# <a id="p-0"></a>Performance
<a id="p-1"></a>Now we execute larger-scale runs to benchmark performance:

<a id="p-2"></a>Code from [PerformanceTester.java:190](https://github.com/SimiaCryptus/mindseye-test/tree/444256810c541076ac97c4437963f8489a719862/src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L190) executed in 0.19 seconds (0.000 gc): 
```java
    test(component, inputPrototype);
```
<a id="p-3"></a>Logging: 
```
    2 batch length, 5 trials
    Input Dimensions:
    	[125, 84, 192]
    Performance:
    	Evaluation performance: 0.009549s +- 0.001216s [0.008325s - 0.011632s]
    	Learning performance: 0.029018s +- 0.001138s [0.027485s - 0.031000s]
    
```
