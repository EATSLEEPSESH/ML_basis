SELECT 
    (SELECT name FROM person WHERE id = po.person_id) AS NAME,
    CASE 
      WHEN (SELECT name FROM person WHERE id = po.person_id) = 'Denis'  
      THEN 'true'  
      ELSE 'false' 
    END as check_name
FROM person_order po
WHERE (menu_id = 13 or menu_id = 14 or menu_id = 18)
  AND order_date = '2022-01-07';