-- Select Data that we are going to use
Select continent,Location, date, total_cases, new_cases, total_deaths, population
From coviddeaths
Order by 1,2;

--BREAKING IT DOWN BY LOCATION (COUNTRY)

-- Looking at Total Cases vs Total Deaths
-- Shows likelihood of dying if you contract COVID in your country
Select continent,location, date, total_cases,total_deaths, (total_deaths/total_cases::float)*100 as DeathPercentage
From coviddeaths
Order by 1,2;

-- Looking at Total Cases vs Population
-- Shows the percentage of the population for the US that has contracted COVID
Select continent,location, date, total_cases, population, (total_cases/population::float)*100 as PercentPopulationInfected
From coviddeaths
Where location= 'United States'
Order by 1,2;

-- Countries with highest infection rate compared to population
Select continent,location, population, max(total_cases) as HighestInfectionCount, max((total_cases/population::float))*100 as PercentPopulationInfected
From coviddeaths
Where total_cases is not NULL and population is not null
Group by location, population
Order by PercentPopulationInfected desc;

-- Countries with highest death count
Select continent,location, max(total_deaths) as TotalDeathCount
From coviddeaths
Where total_deaths is not NULL and continent is not null
Group by location
Order by TotalDeathCount desc;


--BREAKING IT DOWN BY CONTINENT

-- Continents with highest death count
Select location, max(total_deaths) as TotalDeathCount
From coviddeaths
Where continent is null and location not like '%income%'
Group by location
Order by TotalDeathCount desc;


-- GLOBAL NUMBERS

--global death rate by day
Select date, sum(new_cases) as total_cases,sum(new_deaths) as total_deaths, (sum(new_deaths)/sum(new_cases::float))*100 as DeathPercentage
From coviddeaths
Where continent is not null
Group by date
Order by 1,2;

--global death rate
Select sum(new_cases) as total_cases,sum(new_deaths) as total_deaths, (sum(new_deaths)/sum(new_cases::float))*100 as DeathPercentage
From coviddeaths
Where continent is not null
Order by 1,2;

--NOW LOOK AT VACCINATIONS


-- cumulative vaccinations by day by country
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(vac.new_vaccinations) OVER (PARTITION by dea.location Order by dea.date) as Cumulative_Vaccinations
FROM coviddeaths dea
Join covidvaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
Order by 2,3;


-- percent of population vaccinated by day using CTE
With PopvsVac (continent, location, date, population, new_vaccinations, Cumulative_Vaccinations)
as
(

SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(vac.new_vaccinations) OVER (PARTITION by dea.location Order by dea.date) as Cumulative_Vaccinations
FROM coviddeaths dea
Join covidvaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
)
Select *, (Cumulative_Vaccinations/population::float)*100
From PopvsVac;


-- percent of population vaccinated by day using temp table
Drop Table if exists PercentPopulationVaccinated;
Create Table PercentPopulationVaccinated
(
Continent character varying, 
location character varying,
date date,
population bigint,
new_vaccinations int,
Cumulative_Vaccinations real
);

Insert into PercentPopulationVaccinated
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(vac.new_vaccinations) OVER (PARTITION by dea.location Order by dea.date) as Cumulative_Vaccinations
FROM coviddeaths dea
Join covidvaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null;


Select *, (Cumulative_Vaccinations/population::float)*100
From PercentPopulationVaccinated;


-- Creat view to store data for later visualization
Create View ViewPercentPopulationVaccinated as
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(vac.new_vaccinations) OVER (PARTITION by dea.location Order by dea.date) as Cumulative_Vaccinations
FROM coviddeaths dea
Join covidvaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null;

Select *
From ViewPercentPopulationVaccinated;




