# VL-Counter
Zero-shot object counting model with Vison-Language Model


### Requirements
~~~bash
    pip install git+https://github.com/openai/CLIP.git
~~~

### Multiclass Samples
- 336.jpg: blueberries, strawberries
- 343.jpg: kiwis, strawberries


## Experiments

**1. Add CrossEntorpy Loss to density map**
- 물체가 있는 부분을 0으로 예측했을 때 패널티가 큼  
  → 전반적인 값이 높게 예측됨



## 데이터 셋 분석
- 전체 이미지: 6146장
- 평균 객체 갯수: 55.94
- median: 28
- min: 7
- max: 3701

<details>
<summary>객체 별 평균 갯수</summary>
<div markdown="1">
<table>
   <thead>
    <tr style="text-align: right;">
      <th>Name</th>
      <th># of image</th>
      <th>category</th>
      <th>mean</th>
      <th>sum</th>
      <th>min</th>
      <th>max</th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>beads</th><td>423</td><td>train</td><td>64.950</td><td>27474</td><td>7</td><td>428</td><td>59.0</td></tr>
    <tr><th>apples</th><td>221</td><td>test</td><td>68.140</td><td>15059</td><td>8</td><td>675</td><td>38.0</td></tr>
    <tr><th>geese</th><td>216</td><td>train</td><td>20.009</td><td>4322</td><td>8</td><td>89</td><td>15.5</td></tr>
    <tr><th>strawberries</th><td>169</td><td>test</td><td>44.041</td><td>7443</td><td>8</td><td>250</td><td>33.0</td></tr>
    <tr><th>candles</th><td>161</td><td>train</td><td>21.491</td><td>3460</td><td>7</td><td>157</td><td>13.0</td></tr>
    <tr><th>chicken wings</th><td>157</td><td>val</td><td>18.650</td><td>2928</td><td>8</td><td>52</td><td>17.0</td></tr>
    <tr><th>tomatoes</th><td>156</td><td>train</td><td>53.109</td><td>8285</td><td>8</td><td>450</td><td>30.5</td></tr>
    <tr><th>grapes</th><td>155</td><td>val</td><td>73.852</td><td>11447</td><td>11</td><td>757</td><td>53.0</td></tr>
    <tr><th>cranes</th><td>145</td><td>train</td><td>32.269</td><td>4679</td><td>8</td><td>623</td><td>18.0</td></tr>
    <tr><th>bread rolls</th><td>139</td><td>train</td><td>19.014</td><td>2643</td><td>7</td><td>645</td><td>12.0</td></tr>
    <tr><th>cupcake tray</th><td>125</td><td>train</td><td>16.440</td><td>2055</td><td>8</td><td>117</td><td>12.0</td></tr>
    <tr><th>bottle caps</th><td>122</td><td>val</td><td>109.992</td><td>13419</td><td>8</td><td>1229</td><td>50.0</td></tr>
    <tr><th>pigeons</th><td>109</td><td>train</td><td>31.028</td><td>3382</td><td>8</td><td>186</td><td>21.0</td></tr>
    <tr><th>oranges</th><td>103</td><td>train</td><td>67.757</td><td>6979</td><td>8</td><td>438</td><td>48.0</td></tr>
    <tr><th>flamingos</th><td>102</td><td>val</td><td>32.480</td><td>3313</td><td>8</td><td>402</td><td>17.0</td></tr>
    <tr><th>pens</th><td>91</td><td>train</td><td>20.659</td><td>1880</td><td>8</td><td>121</td><td>15.0</td></tr>
    <tr><th>birds</th><td>87</td><td>val</td><td>121.782</td><td>10595</td><td>7</td><td>2092</td><td>47.0</td></tr>
    <tr><th>seagulls</th><td>78</td><td>val</td><td>22.038</td><td>1719</td><td>7</td><td>119</td><td>14.0</td></tr>
    <tr><th>books</th><td>76</td><td>val</td><td>92.803</td><td>7053</td><td>8</td><td>1022</td><td>42.5</td></tr>
    <tr><th>biscuits</th><td>75</td><td>train</td><td>27.520</td><td>2064</td><td>7</td><td>189</td><td>16.0</td></tr>
    <tr><th>chairs</th><td>74</td><td>val</td><td>80.324</td><td>5944</td><td>8</td><td>301</td><td>56.5</td></tr>
    <tr><th>marbles</th><td>73</td><td>test</td><td>51.301</td><td>3745</td><td>8</td><td>275</td><td>33.0</td></tr>
    <tr><th>donuts tray</th><td>73</td><td>val</td><td>13.699</td><td>1000</td><td>8</td><td>54</td><td>12.0</td></tr>
    <tr><th>cups</th><td>71</td><td>train</td><td>44.887</td><td>3187</td><td>9</td><td>405</td><td>28.0</td></tr>
    <tr><th>mini blinds</th><td>71</td><td>train</td><td>39.662</td><td>2816</td><td>9</td><td>196</td><td>32.0</td></tr>
    <tr><th>potatoes</th><td>71</td><td>train</td><td>35.197</td><td>2499</td><td>8</td><td>175</td><td>29.0</td></tr>
    <tr><th>alcohol bottles</th><td>70</td><td>train</td><td>75.814</td><td>5307</td><td>8</td><td>267</td><td>63.0</td></tr>
    <tr><th>crows</th><td>69</td><td>train</td><td>49.087</td><td>3387</td><td>8</td><td>382</td><td>17.0</td></tr>
    <tr><th>green peas</th><td>68</td><td>test</td><td>112.603</td><td>7657</td><td>9</td><td>450</td><td>90.0</td></tr>
    <tr><th>lipstick</th><td>66</td><td>train</td><td>38.333</td><td>2530</td><td>8</td><td>151</td><td>25.5</td></tr>
    <tr><th>stamps</th><td>64</td><td>test</td><td>43.125</td><td>2760</td><td>9</td><td>216</td><td>32.5</td></tr>
    <tr><th>sunglasses</th><td>63</td><td>test</td><td>66.206</td><td>4171</td><td>8</td><td>253</td><td>47.0</td></tr>
    <tr><th>eggs</th><td>63</td><td>test</td><td>32.825</td><td>2068</td><td>8</td><td>145</td><td>24.0</td></tr>
    <tr><th>bricks</th><td>58</td><td>train</td><td>270.690</td><td>15700</td><td>12</td><td>1912</td><td>143.5</td></tr>
    <tr><th>coins</th><td>58</td><td>train</td><td>69.069</td><td>4006</td><td>8</td><td>433</td><td>54.5</td></tr>
    <tr><th>pencils</th><td>58</td><td>train</td><td>39.897</td><td>2314</td><td>11</td><td>274</td><td>26.0</td></tr>
    <tr><th>macarons</th><td>58</td><td>train</td><td>22.345</td><td>1296</td><td>9</td><td>229</td><td>16.0</td></tr>
    <tr><th>coffee beans</th><td>55</td><td>train</td><td>86.691</td><td>4768</td><td>9</td><td>504</td><td>82.0</td></tr>
    <tr><th>pills</th><td>55</td><td>val</td><td>52.673</td><td>2897</td><td>9</td><td>283</td><td>28.0</td></tr>
    <tr><th>kidney beans</th><td>54</td><td>train</td><td>73.426</td><td>3965</td><td>8</td><td>274</td><td>54.5</td></tr>
    <tr><th>polka dots</th><td>53</td><td>val</td><td>181.302</td><td>9609</td><td>9</td><td>885</td><td>100.0</td></tr>
    <tr><th>nuts</th><td>53</td><td>train</td><td>51.887</td><td>2750</td><td>8</td><td>159</td><td>42.0</td></tr>
    <tr><th>cashew nuts</th><td>51</td><td>test</td><td>48.118</td><td>2454</td><td>11</td><td>193</td><td>43.0</td></tr>
    <tr><th>finger foods</th><td>50</td><td>test</td><td>21.040</td><td>1052</td><td>8</td><td>189</td><td>13.5</td></tr>
    <tr><th>cars</th><td>48</td><td>train</td><td>110.562</td><td>5307</td><td>8</td><td>485</td><td>67.0</td></tr>
    <tr><th>cans</th><td>48</td><td>train</td><td>85.688</td><td>4113</td><td>8</td><td>1672</td><td>37.0</td></tr>
    <tr><th>windows</th><td>48</td><td>train</td><td>62.875</td><td>3018</td><td>12</td><td>381</td><td>43.5</td></tr>
    <tr><th>pearls</th><td>45</td><td>train</td><td>73.356</td><td>3301</td><td>10</td><td>413</td><td>41.0</td></tr>
    <tr><th>jade stones</th><td>45</td><td>train</td><td>25.911</td><td>1166</td><td>10</td><td>88</td><td>22.0</td></tr>
    <tr><th>shoes</th><td>44</td><td>train</td><td>46.159</td><td>2031</td><td>8</td><td>239</td><td>35.0</td></tr>
    <tr><th>cupcakes</th><td>44</td><td>train</td><td>23.205</td><td>1021</td><td>8</td><td>123</td><td>15.5</td></tr>
    <tr><th>caps</th><td>43</td><td>train</td><td>62.023</td><td>2667</td><td>14</td><td>193</td><td>51.0</td></tr>
    <tr><th>balls</th><td>40</td><td>train</td><td>40.425</td><td>1617</td><td>8</td><td>326</td><td>16.5</td></tr>
    <tr><th>roof tiles</th><td>39</td><td>train</td><td>155.821</td><td>6077</td><td>16</td><td>815</td><td>109.0</td></tr>
    <tr><th>stairs</th><td>38</td><td>train</td><td>16.816</td><td>639</td><td>9</td><td>64</td><td>13.5</td></tr>
    <tr><th>keyboard keys</th><td>37</td><td>test</td><td>103.568</td><td>3832</td><td>42</td><td>409</td><td>86.0</td></tr>
    <tr><th>comic books</th><td>37</td><td>test</td><td>49.162</td><td>1819</td><td>12</td><td>235</td><td>36.0</td></tr>
    <tr><th>plates</th><td>36</td><td>train</td><td>49.167</td><td>1770</td><td>8</td><td>628</td><td>22.5</td></tr>
    <tr><th>peaches</th><td>36</td><td>val</td><td>42.667</td><td>1536</td><td>8</td><td>133</td><td>34.5</td></tr>
    <tr><th>buns</th><td>35</td><td>train</td><td>15.029</td><td>526</td><td>8</td><td>47</td><td>12.0</td></tr>
    <tr><th>markers</th><td>34</td><td>test</td><td>246.676</td><td>8387</td><td>9</td><td>3701</td><td>102.0</td></tr>
    <tr><th>tree logs</th><td>32</td><td>test</td><td>88.219</td><td>2823</td><td>15</td><td>363</td><td>53.5</td></tr>
    <tr><th>nail polish</th><td>30</td><td>test</td><td>119.367</td><td>3581</td><td>24</td><td>508</td><td>91.0</td></tr>
    <tr><th>bowls</th><td>30</td><td>train</td><td>34.133</td><td>1024</td><td>9</td><td>92</td><td>22.5</td></tr>
    <tr><th>spoon</th><td>29</td><td>train</td><td>19.310</td><td>560</td><td>8</td><td>49</td><td>16.0</td></tr>
    <tr><th>cement bags</th><td>27</td><td>train</td><td>54.519</td><td>1472</td><td>10</td><td>170</td><td>41.0</td></tr>
    <tr><th>elephants</th><td>26</td><td>test</td><td>25.154</td><td>654</td><td>10</td><td>107</td><td>19.5</td></tr>
    <tr><th>kiwis</th><td>25</td><td>val</td><td>29.360</td><td>734</td><td>8</td><td>87</td><td>24.0</td></tr>
    <tr><th>go game</th><td>24</td><td>train</td><td>140.750</td><td>3378</td><td>10</td><td>276</td><td>143.5</td></tr>
    <tr><th>sheep</th><td>24</td><td>test</td><td>110.708</td><td>2657</td><td>20</td><td>209</td><td>113.0</td></tr>
    <tr><th>boxes</th><td>24</td><td>train</td><td>79.708</td><td>1913</td><td>17</td><td>219</td><td>67.0</td></tr>
    <tr><th>toilet paper rolls</th><td>23</td><td>val</td><td>62.217</td><td>1431</td><td>8</td><td>907</td><td>18.0</td></tr>
    <tr><th>fishes</th><td>23</td><td>train</td><td>53.261</td><td>1225</td><td>8</td><td>303</td><td>28.0</td></tr>
    <tr><th>cotton balls</th><td>23</td><td>train</td><td>27.304</td><td>628</td><td>8</td><td>69</td><td>22.0</td></tr>
    <tr><th>horses</th><td>23</td><td>val</td><td>17.609</td><td>405</td><td>8</td><td>65</td><td>13.0</td></tr>
    <tr><th>ants</th><td>22</td><td>val</td><td>37.409</td><td>823</td><td>8</td><td>238</td><td>19.5</td></tr>
    <tr><th>penguins</th><td>21</td><td>train</td><td>25.571</td><td>537</td><td>8</td><td>125</td><td>13.0</td></tr>
    <tr><th>shirts</th><td>20</td><td>val</td><td>137.600</td><td>2752</td><td>9</td><td>1231</td><td>38.0</td></tr>
    <tr><th>hot air balloons</th><td>20</td><td>test</td><td>22.900</td><td>458</td><td>8</td><td>113</td><td>12.0</td></tr>
    <tr><th>buffaloes</th><td>19</td><td>train</td><td>106.053</td><td>2015</td><td>9</td><td>298</td><td>60.0</td></tr>
    <tr><th>people</th><td>19</td><td>train</td><td>71.053</td><td>1350</td><td>23</td><td>226</td><td>50.0</td></tr>
    <tr><th>cassettes</th><td>19</td><td>train</td><td>58.105</td><td>1104</td><td>9</td><td>146</td><td>54.0</td></tr>
    <tr><th>skateboard</th><td>19</td><td>val</td><td>36.263</td><td>689</td><td>8</td><td>127</td><td>26.0</td></tr>
    <tr><th>deers</th><td>19</td><td>test</td><td>26.368</td><td>501</td><td>9</td><td>77</td><td>21.0</td></tr>
    <tr><th>watches</th><td>18</td><td>test</td><td>39.722</td><td>715</td><td>12</td><td>111</td><td>29.5</td></tr>
    <tr><th>jeans</th><td>18</td><td>train</td><td>37.111</td><td>668</td><td>10</td><td>113</td><td>21.0</td></tr>
    <tr><th>cereals</th><td>18</td><td>train</td><td>26.167</td><td>471</td><td>18</td><td>43</td><td>25.0</td></tr>
    <tr><th>boats</th><td>17</td><td>train</td><td>56.765</td><td>965</td><td>8</td><td>145</td><td>35.0</td></tr>
    <tr><th>cows</th><td>16</td><td>train</td><td>22.375</td><td>358</td><td>13</td><td>50</td><td>21.0</td></tr>
    <tr><th>sauce bottles</th><td>15</td><td>test</td><td>34.667</td><td>520</td><td>14</td><td>68</td><td>34.0</td></tr>
    <tr><th>ice cream</th><td>15</td><td>train</td><td>33.000</td><td>495</td><td>10</td><td>110</td><td>24.0</td></tr>
    <tr><th>legos</th><td>14</td><td>test</td><td>303.000</td><td>4242</td><td>13</td><td>2560</td><td>90.5</td></tr>
    <tr><th>mosaic tiles</th><td>14</td><td>train</td><td>149.929</td><td>2099</td><td>18</td><td>953</td><td>52.5</td></tr>
    <tr><th>milk cartons</th><td>14</td><td>val</td><td>72.143</td><td>1010</td><td>22</td><td>185</td><td>66.0</td></tr>
    <tr><th>zebras</th><td>14</td><td>train</td><td>55.286</td><td>774</td><td>10</td><td>191</td><td>31.5</td></tr>
    <tr><th>bananas</th><td>14</td><td>train</td><td>41.143</td><td>576</td><td>8</td><td>115</td><td>27.5</td></tr>
    <tr><th>camels</th><td>14</td><td>val</td><td>18.929</td><td>265</td><td>8</td><td>58</td><td>14.0</td></tr>
    <tr><th>carrom board pieces</th><td>13</td><td>test</td><td>20.077</td><td>261</td><td>13</td><td>29</td><td>19.0</td></tr>
    <tr><th>peppers</th><td>13</td><td>train</td><td>18.692</td><td>243</td><td>10</td><td>47</td><td>13.0</td></tr>
    <tr><th>cartridges</th><td>12</td><td>train</td><td>79.167</td><td>950</td><td>33</td><td>179</td><td>56.5</td></tr>
    <tr><th>flower pots</th><td>12</td><td>val</td><td>41.500</td><td>498</td><td>18</td><td>76</td><td>32.5</td></tr>
    <tr><th>fresh cut</th><td>11</td><td>val</td><td>81.818</td><td>900</td><td>20</td><td>177</td><td>64.0</td></tr>
    <tr><th>sticky notes</th><td>11</td><td>test</td><td>71.636</td><td>788</td><td>29</td><td>121</td><td>60.0</td></tr>
    <tr><th>watermelon</th><td>11</td><td>train</td><td>43.909</td><td>483</td><td>9</td><td>149</td><td>27.0</td></tr>
    <tr><th>bullets</th><td>11</td><td>val</td><td>19.727</td><td>217</td><td>8</td><td>40</td><td>19.0</td></tr>
    <tr><th>skis</th><td>11</td><td>test</td><td>14.727</td><td>162</td><td>8</td><td>26</td><td>13.0</td></tr>
    <tr><th>red beans</th><td>10</td><td>test</td><td>78.500</td><td>785</td><td>15</td><td>161</td><td>80.5</td></tr>
    <tr><th>goldfish snack</th><td>10</td><td>train</td><td>37.300</td><td>373</td><td>8</td><td>103</td><td>21.5</td></tr>
    <tr><th>crayons</th><td>10</td><td>train</td><td>21.900</td><td>219</td><td>8</td><td>64</td><td>12.0</td></tr>
    <tr><th>stapler pins</th><td>9</td><td>train</td><td>51.000</td><td>459</td><td>15</td><td>98</td><td>50.0</td></tr>
    <tr><th>clams</th><td>9</td><td>train</td><td>43.222</td><td>389</td><td>10</td><td>190</td><td>22.0</td></tr>
    <tr><th>oyster shells</th><td>9</td><td>val</td><td>18.778</td><td>169</td><td>9</td><td>49</td><td>12.0</td></tr>
    <tr><th>gemstones</th><td>8</td><td>train</td><td>65.250</td><td>522</td><td>8</td><td>393</td><td>11.0</td></tr>
    <tr><th>matches</th><td>8</td><td>train</td><td>20.375</td><td>163</td><td>10</td><td>60</td><td>12.0</td></tr>
    <tr><th>straws</th><td>7</td><td>train</td><td>68.429</td><td>479</td><td>19</td><td>113</td><td>64.0</td></tr>
    <tr><th>m&m pieces</th><td>7</td><td>train</td><td>45.143</td><td>316</td><td>10</td><td>111</td><td>29.0</td></tr>
    <tr><th>spring rolls</th><td>7</td><td>train</td><td>16.429</td><td>115</td><td>8</td><td>31</td><td>10.0</td></tr>
    <tr><th>candy pieces</th><td>6</td><td>test</td><td>36.000</td><td>216</td><td>8</td><td>111</td><td>25.0</td></tr>
    <tr><th>onion rings</th><td>6</td><td>train</td><td>24.333</td><td>146</td><td>8</td><td>50</td><td>20.5</td></tr>
    <tr><th>swans</th><td>6</td><td>train</td><td>21.500</td><td>129</td><td>8</td><td>48</td><td>17.0</td></tr>
    <tr><th>chopstick</th><td>6</td><td>train</td><td>21.000</td><td>126</td><td>12</td><td>36</td><td>19.0</td></tr>
    <tr><th>bees</th><td>5</td><td>train</td><td>40.600</td><td>203</td><td>22</td><td>59</td><td>37.0</td></tr>
    <tr><th>sausages</th><td>5</td><td>val</td><td>35.800</td><td>179</td><td>10</td><td>104</td><td>25.0</td></tr>
    <tr><th>baguette rolls</th><td>5</td><td>train</td><td>21.200</td><td>106</td><td>10</td><td>53</td><td>15.0</td></tr>
    <tr><th>chewing gum pieces</th><td>5</td><td>train</td><td>16.200</td><td>81</td><td>10</td><td>31</td><td>14.0</td></tr>
    <tr><th>croissants</th><td>5</td><td>train</td><td>13.800</td><td>69</td><td>8</td><td>33</td><td>9.0</td></tr>
    <tr><th>potato chips</th><td>4</td><td>test</td><td>25.500</td><td>102</td><td>11</td><td>36</td><td>27.5</td></tr>
    <tr><th>sea shells</th><td>4</td><td>test</td><td>20.250</td><td>81</td><td>8</td><td>36</td><td>18.5</td></tr>
    <tr><th>shallots</th><td>4</td><td>val</td><td>18.750</td><td>75</td><td>7</td><td>31</td><td>18.5</td></tr>
    <tr><th>crab cakes</th><td>4</td><td>test</td><td>14.750</td><td>59</td><td>10</td><td>20</td><td>14.5</td></tr>
    <tr><th>oysters</th><td>4</td><td>val</td><td>10.500</td><td>42</td><td>8</td><td>16</td><td>9.0</td></tr>
    <tr><th>naan bread</th><td>3</td><td>train</td><td>29.333</td><td>88</td><td>19</td><td>35</td><td>34.0</td></tr>
    <tr><th>instant noodles</th><td>3</td><td>train</td><td>18.000</td><td>54</td><td>12</td><td>28</td><td>14.0</td></tr>
    <tr><th>birthday candles</th><td>3</td><td>train</td><td>15.667</td><td>47</td><td>11</td><td>23</td><td>13.0</td></tr>
    <tr><th>meat skewers</th><td>3</td><td>train</td><td>8.333</td><td>25</td><td>8</td><td>9</td><td>8.0</td></tr>
    <tr><th>goats</th><td>2</td><td>train</td><td>50.000</td><td>100</td><td>11</td><td>89</td><td>50.0</td></tr>
    <tr><th>rice bags</th><td>2</td><td>train</td><td>38.000</td><td>76</td><td>26</td><td>50</td><td>38.0</td></tr>
    <tr><th>flowers</th><td>2</td><td>val</td><td>31.000</td><td>62</td><td>20</td><td>42</td><td>31.0</td></tr>
    <tr><th>kitchen towels</th><td>2</td><td>train</td><td>18.500</td><td>37</td><td>9</td><td>28</td><td>18.5</td></tr>
    <tr><th>calamari rings</th><td>2</td><td>train</td><td>15.500</td><td>31</td><td>9</td><td>22</td><td>15.5</td></tr>
    <tr><th>prawn crackers</th><td>2</td><td>val</td><td>10.500</td><td>21</td><td>8</td><td>13</td><td>10.5</td></tr>
    <tr><th>nails</th><td>1</td><td>train</td><td>115.000</td><td>115</td><td>115</td><td>115</td><td>115.0</td></tr>
    <tr><th>polka dot tiles</th><td>1</td><td>train</td><td>107.000</td><td>107</td><td>107</td><td>107</td><td>107.0</td></tr>
    <tr><th>screws</th><td>1</td><td>train</td><td>100.000</td><td>100</td><td>100</td><td>100</td><td>100.0</td></tr>
    <tr><th>lighters</th><td>1</td><td>train</td><td>50.000</td><td>50</td><td>50</td><td>50</td><td>50.0</td></tr>
    <tr><th>bottles</th><td>1</td><td>train</td><td>12.000</td><td>12</td><td>12</td><td>12</td><td>12.0</td></tr>
    <tr><th>supermarket shelf</th><td>1</td><td>train</td><td>8.000</td><td>8</td><td>8</td><td>8</td><td>8.0</td></tr>
  </tbody>
</table>
</div>
</details>
