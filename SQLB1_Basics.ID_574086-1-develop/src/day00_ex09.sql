SELECT (SELECT name FROM person
WHERE PV.person_id = id) AS person_name,
(SELECT name FROM pizzeria
WHERE PV.pizzeria_id = id) AS pizzeria_name
FROM (SELECT person_id, pizzeria_id FROM person_visits
WHERE visit_date BETWEEN '2022-01-07' AND '2022-01-09') AS PV 
ORDER BY person_name, pizzeria_name DESC