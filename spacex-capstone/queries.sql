-- 01_unique_sites
SELECT DISTINCT Launch_Site FROM SPACEX ORDER BY Launch_Site;

-- 02_cca_records
SELECT Date, Booster_Version, Launch_Site FROM SPACEX WHERE Launch_Site LIKE 'CCA%' ORDER BY Date LIMIT 5;

-- 03_nasa_payload
SELECT SUM(PAYLOAD_MASS__KG_) AS TotalPayloadKg FROM SPACEX WHERE Customer LIKE '%NASA%';

-- 04_v11_average
SELECT AVG(PAYLOAD_MASS__KG_) AS AveragePayloadKg FROM SPACEX WHERE Booster_Version='F9 v1.1';

-- 05_first_ground
SELECT MIN(Date) AS FirstGroundLanding FROM SPACEX WHERE Landing_Outcome='Success (ground pad)';

-- 06_drone_payload
SELECT Date,Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEX WHERE Landing_Outcome='Success (drone ship)' AND PAYLOAD_MASS__KG_>4000 AND PAYLOAD_MASS__KG_<6000 ORDER BY Date;

-- 07_mission_outcomes
SELECT Mission_Outcome,COUNT(*) AS Launches FROM SPACEX GROUP BY Mission_Outcome;

-- 08_max_payload
SELECT Date,Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEX WHERE PAYLOAD_MASS__KG_=(SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEX);

-- 09_2015_failures
SELECT Date,Booster_Version,Launch_Site FROM SPACEX WHERE Date LIKE '2015%' AND Landing_Outcome='Failure (drone ship)';

-- 10_rank_outcomes
SELECT Landing_Outcome,COUNT(*) AS Launches FROM SPACEX WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Landing_Outcome ORDER BY Launches DESC;