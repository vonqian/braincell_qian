# Basket Cell (BC) 离子通道分布与电生理参数汇总

## 1. 神经元各区域反转电位参数表

| 参数名称  | 参数值   | 单位  |
| ----- | ----- | --- |
| Ena   | 60.0  | mV  |
| Ek    | -80.0 | mV  |
| Eh    | -34.0 | mV  |
| Eca   | 137.5 | mV  |
| Eleak | -55.0 | mV  |


---

## 2. 神经元各区域离子通道参数总表

|所属区域|通道名称|参数名称|参数值|单位|
|---|---|---|---|---|
|Soma|Leak|gmax|4.00E-05|S/cm²|
||Nav1_1|gbar|0.200000|S/cm²|
||Cav3_2|gcabar|1.00E-04|S/cm²|
||Cav12|gbar|7.00E-04|S/cm²|
||Cav13|gbar|5.00E-06|S/cm²|
||Kir2_3|gkbar|1.00E-04|S/cm²|
||Kv3_4|gkbar|0.097000|S/cm²|
||Kv4_3|gkbar|0.010000|S/cm²|
||Kca3_1|gkbar|0.001000|S/cm²|
||HCN1_PC|gbar|0.001000|S/cm²|
||cdp5StCmod|TotalPump|2.00E-09|mol/cm²/s|
|Dendrite|Leak|gmax|1.00E-05|S/cm²|
||Cav3_2|gcabar|5.00E-05|S/cm²|
||Cav12|gbar|2.00E-04|S/cm²|
||Cav13|gbar|5.00E-06|S/cm²|
||Kv4_3|gkbar|0.009872|S/cm²|
||Kca2_2|gkbar|0.006500|S/cm²|
||cdp5StCmod|TotalPump|2.00E-09|mol/cm²/s|
|AIS (axon[0])|Leak|gmax|1.00E-05|S/cm²|
||Nav1_6|gbar|0.300000|S/cm²|
||Kv3_4|gkbar|0.002000|S/cm²|
||HCN1_PC|gbar|0.001000|S/cm²|
||Kca1_1|gkbar|0.010000|S/cm²|
||Cav2_1|gcabar|2.20E-04|S/cm²|
||cdp5StCmod|TotalPump|2.00E-09|mol/cm²/s|
|Axon (axon[1:])|Leak|gmax|1.00E-06|S/cm²|
||Nav1_6|gbar|0.001000|S/cm²|
||Kv3_4|gkbar|0.001000|S/cm²|
||Kv1_1|gkbar|5.00E-04|S/cm²|
||HCN1_PC|gbar|1.00E-04|S/cm²|
||Kca1_1|gkbar|0.001000|S/cm²|
||Cav2_1|gcabar|8.00E-05|S/cm²|
||cdp5StCmod|TotalPump|2.00E-09|mol/cm²/s|

---

## 3. 离子通道空间分布矩阵

说明：

- **●** ：该区域表达该通道
    
- **–** ：该区域未表达该通道
    

|通道|Soma|Dendrite|AIS (axon[0])|Axon (axon[1:])|
|---|---|---|---|---|
|Leak|●|●|●|●|
|Nav1.1|●|–|–|–|
|Nav1.6|–|–|●|●|
|Cav3.2|●|●|–|–|
|Cav1.2|●|●|–|–|
|Cav1.3|●|●|–|–|
|Cav2.1|–|–|●|●|
|Kir2.3|●|–|–|–|
|Kv3.4|●|–|●|●|
|Kv4.3|●|●|–|–|
|Kv1.1|–|–|–|●|
|Kca3.1|●|–|–|–|
|Kca2.2|–|●|–|–|
|Kca1.1|–|–|●|●|
|HCN1|●|–|●|●|
|Ca Pump|●|●|●|●|

---

