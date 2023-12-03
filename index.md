# What is Food Insecurity 
According to the USDA, food insecurity is defined as the “lack of consistent access to enough food for every person in a household to live an active, healthy life.” Food insecurity can be caused by several different life circumstances including poverty, illness, and systemic racism [1]. 

# Why is it Important?
Food insecurity is widespread across the United States, affecting more than 44 million people, 13 million of those being children [1]. In Georgia alone, 1 in 9 people face food insecurity [2]. Food insecurity deeply affects people’s daily lives, and prohibits the ability to live a full, healthy life. Because it is such a vast issue, it is also a difficult one to solve.

<img src="images/num_insecure.png" width="80%" />

<!--- ![image1](/images/num_insecure.png) ---> 

# How Can We Help?
A key step to addressing the issue of food insecurity is raising awareness. This project aims to explore the issue of food insecurity here in Georgia, highlighting where there may be food deserts within the state. We aim to explore food insecurity in each county by different variables, demonstrating possible predictors of food insecurity. Our goal in doing so is to provide you with knowledge on this issue, so that we as a community can better advocate for reform and inform others about food insecurity in our own communities. 

# Terminology to Know
**Low-Access:** In this dataset, food access can be defined by  the distance to the nearest supermarket, supercenter, or large grocery store. For urban counties, we consider individuals who live more than 1 mile away from the nearest supermarket or grocery store as food insecure. For rural counties, where population density is intrinsically lower and grocery stores are fewer, individuals are considered to be food insecure if they live greater than 10 miles from the nearest food source.  

**Low-Income:** Low-income counties, as defined by the Department of Treasury’s New Markets Tax Credit (NMTC) program, are those where the poverty rate is 20% or greater, the median family income of the county is less than 80% of the state’s median family income, or the tract is in a metropolitan area and the median family income is less than 80% of the greater metropolitan median family income. 

**Food Desert:** Regions of the country [that] often feature large proportions of households with low incomes, inadequate access to transportation, and a limited number of food retailers providing fresh produce and healthy groceries for affordable prices. [6]

**Poverty Rate:** The proportion of the population in the county that lives below the Federal poverty threshold. 

**Urban:** According to the US Census, to qualify as an urban area, the territory identified according to criteria must encompass at least 2,000 housing units or have a population of at least 5,000.

**Rural:** According to the US Census, all people, housing, and territory that are not within an urban area.

**SNAP:** Supplemental Nutrition Assistance Program (SNAP) is the largest federal nutrition assistance program. SNAP provides benefits to eligible low-income individuals and families via an Electronic Benefits Transfer card. This card can be used like a debit card to purchase eligible food in authorized retail food stores. [5]

**Tract:** small, statistical subdivisions of a county used by the US Census to gather data. For our project, tract data was aggregated to be cumulative data sets of the entire county, rather than individual tracts.

# Food Insecurity by County
Georgia is made up of 159 counties. The counties can be differentiated as rural and urban. 
The graph below outlines the counties in Georgia that can be identified as rural and urban. As you can see…

[INSERT GRAPH]

**Does living in a rural vs. urban area affect food insecurity in each county?**

<img src="images/num_rural_urban.png" width="80%">

<!--- ![image2](/images/num_rural_urban.png) ---> 

The graph below shows the percent of county populations that are low income and low access, two factors that determine food insecurity. Low access and low income are encoded by saturation of color (darker being higher percentages of low income/access, lighter being lower percentages). Rural and Urban are determined by specific colors (blue for rural and brown for urban).

Taliaferro County, a rural county east of Atlanta, has the highest percentage of food insecurity. Among its 1,717 residents, 47.43% are low income and have low access to food. Chattahoochee County, an urban county southeast of Columbus, has the second highest percentage of food insecurity, as 39.34% of its 11,267 residents are low income and have low access to food. In Georgia, urban residents are more likely to face food insecurity. 11.11% of the urban population is low income and has low access to food, compared to 7.15% of the rural population.

[INSERT GRAPH]

# Food Insecurity by Distance

# Food Insecurity by Population Density
Living in a rural or urban area can greatly affect your proximity to a viable food source. The Pareto chart below explores the question: **How does a county’s population/population density influence the likelihood that the county has food insecurity issues?** The histogram bins counties by their population density. The color saturation/heat map encodes the number of food insecurity in the county at 1 and 10 miles (these are Census identified mile markers for food deserts). The orange trend line shows the cumulative percentage of food insecurity cases across the state , and allows you to see which levels of  population density account for what percentage of food insecurity instances in the state. Looking at the graphs, we can see that there is a large distinction between population density in rural and urban areas. This gap indicates that areas in Georgia are either very sparsely populated or very heavily populated, with little existence of a middle ground. While you might expect that food insecurity is more present in sparsely populated areas, where people would theoretically be living further from grocery stores, the opposite is actually true. In fact ~38% of the food insecurity cases in Georgia exist in the 5 most densely populated counties: Fulton, Clayton, Gwinnett, Cobb, and DeKalb. When considering just counties that contain more urban census tracts than rural census tracts, their responsibility for food insecurity cases jumps to around 45%. 

[INSERT GRAPH]

# Food Insecurity by Vehicle Access

# Food Insecurity by Race
Systemic racism is a big factor in the creation of communities with high levels of food insecurity. The following graph demonstrates food insecurity by race. Each dot is a county, and the x-axis shows the low access population of each race for a particular county. The size of the dot corresponds to the size of the total low access population of a county. As we expect, larger counties tend to have larger low access populations for each race; however, some counties, such as Cherokee county, have a large low access population for a particular race despite not having a large total low access population. This suggests that there is a disparity in food insecurity for certain races in certain counties.To compare the low access populations by race for a particular county, click on one of the dots, and the visualization will highlight all of the dots for that county.

<div style="margin:1em calc(50% - 50vw);">
        <div class='tableauPlaceholder' id='viz1701576951539' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 '           
                src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book2_17014750715760&#47;Dashboard1&#47;1_rss.png'
                style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'>
          <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
          <param name='embed_code_version' value='3' />
          <param name='site_root' value='' />
          <param name='name' value='Book2_17014750715760&#47;Dashboard1' />
          <param name='tabs' value='no' /><param name='toolbar' value='yes' />
          <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book2_17014750715760&#47;Dashboard1&#47;1.png' />
          <param name='animate_transition' value='yes' />
          <param name='display_static_image' value='yes' />
          <param name='display_spinner' value='yes' />
          <param name='display_overlay' value='yes' />
          <param name='display_count' value='yes' />
          <param name='language' value='en-US' />
        </object></div>                
        <script type='text/javascript'>                    var divElement = document.getElementById('viz1701576951539');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);          </script>
</div>

s

<div style="margin:1em calc(50% - 50vw);">
<div class='tableauPlaceholder' id='viz1701580133476' style='position: relative'><noscript><a href='#'><img alt='Sheet 2 (2) ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book1_17013902158690&#47;Sheet22&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book1_17013902158690&#47;Sheet22' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book1_17013902158690&#47;Sheet22&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1701580133476');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
</div>

# Food Insecurity in Seniors vs Children
In addition to race, seniors and children experience high levels of food insecurity. Seniors in particular generally have limited finances and resources [3]. Children experience food insecurity at high rates, especially those situated in single mother households, which has one of the highest food insecurity rates [4]. The unit visualizations below show the number of kids and seniors in each county that face food insecurity issues. Each symbol represents 1000 people, and the counties are represented by color. The biggest takeaway from this graph is the sheer number of these more vulnerable groups, and especially children, that experience food insecurity. Again, we see the most densely populated counties experiencing extreme amounts of food insecurity, with Gwinnett alone having 93,063 children experiencing food insecurity issues. Generally speaking, the five most heavily populated counties again top the charts with cases for vulnerable groups, though not necessarily in the same order.

[INSERT GRAPH]

# Food Insecurity by Governmental Benefits
**Does a county’s number of SNAP recipients and the poverty rate have any correlation?**

Governmental assistance programs such as SNAP are a viable resource to provide people with greater access to food. The following scatterplot demonstrates poverty rate on the x axis, and allows for you to change the y axis to SNAP households in general, and SNAP households that are either 1 mile or 10 miles from a grocery store (these are two distance standards for food deserts for urban and rural areas respectively, as determined by the US Census). The scatterplots reveal that though there are many households that utilize SNAP in Georgia, there is not much correlation between poverty rate and SNAP usage or lack thereof, even for those living within the furthest measured distance from a grocery store (10 miles). 

[INSERT GRAPH]

# Close to Home: Metro ATL Counties Food Insecurity Ranking
Now that we know more about the prevalence of food insecurity in Georgia, you may be overwhelmed by the question “There’s so much need across the state, how can we help?”. In order to put the need into a more tangible context, the bump chart below shows the counties where we are, in the Metro Atlanta area, ranked from 1-11, 1 being the most food insecure of the group, and 11 being the least, across 4 years from 2018-2021 [7]. Hover over each point to see the percentage of food insecurity in each county over the years. As we can see, food insecurity is an issue closer to us than it seems. So, what can we do about it?

[INSERT GRAPH]

There are many things we can do locally to take action towards solving food insecurity:

-[Volunteer](https://www.acfb.org/volunteer/) at your local food bank. The Atlanta Community Food Bank serves these counties and more, and has several volunteer opportunities for you to get involved and make a difference in your local community.

-Donate to organizations who are fighting hunger and food insecurity. 

-Continue to raise awareness of food insecurity! Food insecurity affects people all across the country. The more informed we are about the issue, the better equipped we will be to solve it.

# About the Data
Data and information is sourced from the 2010, 2015, and 2019 Food Access Research Atlas Data provided by the USDA Economic Research Service. 

<https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/>

Data for the bump chart that outlined counties’ ranked food insecurity from the years 2018- 2021 is sourced from data provided by Feeding America.

<https://map.feedingamerica.org/county/2021/overall/georgia/county/rockdale>

# Sources
[1] <https://www.feedingamerica.org/hunger-in-america/food-insecurity>

[2] <https://www.feedingamerica.org/hunger-in-america/georgia>

[3] <https://frac.org/wp-content/uploads/hunger-is-a-health-issue-for-older-adults-1.pdf>

[4] <https://www.ers.usda.gov/data-products/ag-and-food-statistics-charting-the-essentials/food-security-and-nutrition-assistance/#:~:text=Food%20insecurity%20rates%20are%20highest,and%20very%20low%20food%20security.>

[5] <https://www.benefits.gov/benefit/361>

[6] <https://www.ers.usda.gov/webdocs/publications/45014/30940_err140.pdf>

[7] <https://map.feedingamerica.org/county/2021/overall/georgia/county/rockdale>
