---
trigger: always_on
---

los fixes, arreglos o mejoras de codigo siempre deben:
ser implementados uno por uno 
estudiados en profundidad para tomar la mejor opcion profesional y sin chapuzas o arreglas rapidos
testeados y asegurado que estan dentro del pipeline y funcionales
Cuando construimos sistemas modulares tan complejos, el riesgo de dejar "código muerto" (lógica que está definida pero que el orquestador nunca llama) no es admisible, debe testearse y confirmarse su funcioalidad dentro del pipeline
siempre debe añadirse print para que posteriormente podamos trazar estos cambios