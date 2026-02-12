from src.cover_letter_generator import CoverLetterGenerator
from src.rag_chain import RAGChain
from src.vector_store import VectorStoreManager

def main():
    """Main function to generate a cover letter"""
    
    # Job description - paste the full job posting here
    job_description = """
Consultor Ingeniero de Datos (IA)

Ubicación: Colombia
Modalidad: Remoto desde Colombia

Estamos buscando un Ingeniero de Datos con al menos 3 años de experiencia, para apoyar procesos ETL, manejo y calidad de datos, y mantenimiento de flujos existentes.


Tu misión principal será garantizar el flujo, la calidad y la disponibilidad de los datos, participando activamente en el mantenimiento de flujos actuales (On-Premise) y apoyando la evolución hacia la nube


Responsabilidades:

Desarrollo y Mantenimiento de ETLs: Diseñar, construir y mantener procesos de extracción, transformación y carga (ETL) utilizando Pentaho Data Integration (PDI/Kettle).
Gestión de Data Warehouse: Administrar consultas y optimizar el rendimiento de bases de datos masivas en Teradata.
Integración Cloud: Colaborar en la ingesta y migración de datos hacia AWS (S3, Redshift, Glue), asegurando la integridad de la información durante el proceso.
Modelado y Calidad de Datos: Asegurar que las tablas y estructuras de datos estén limpias, documentadas y listas para el consumo de los equipos de analítica.
Resolución de Incidentes: Monitorear los "jobs" diarios y solucionar fallos en la carga de datos de manera oportuna.


Requisitos:

Experiencia: Mínimo 2 años comprobables en roles de Ingeniería de Datos o BI.
Dominio de SQL: Nivel avanzado (Joins complejos, Stored Procedures, optimización de queries).
Herramientas ETL: Experiencia sólida con Pentaho (PDI).
Bases de Datos: Experiencia trabajando con Teradata (o bases de datos MPP similares).
Nube: Conocimientos prácticos de AWS (conceptos básicos de S3, IAM, y servicios de datos).


¿Qué ofrecemos?



Modalidad: 100% Remoto
Contrato: Indefinido
Beneficios: Medicina prepagada, plan odontológico, descuentos corporativos y más.




Te encantará trabajar en Capgemini porque:



· Ofrecemos una experiencia única de reclutamiento y onboarding, y te ayudamos a construir las bases de tu carrera y habilidades profesionales.
· Proveemos un ambiente de trabajo colaborativo basado en nuestros 7 valores: Honestidad, Audacia, Confianza, Libertad, Espíritu de Equipo, Modestia y Diversión.
· Promovemos un ambiente que te permite planear y desarrollar tu carrera.


Aplica si tienes el perfil requerido.



En Capgemini Colombia buscamos atraer al mejor talento y estamos comprometidos con la creación de un ambiente de trabajo diverso e inclusivo, para que no exista discriminación por motivos de raza, sexo, orientación sexual, identidad, expresión de género o cualquier otra característica de una persona. Todas las solicitudes son bienvenidas y se considerarán en función del mérito para el trabajo y/o la experiencia para el puesto.
    """
    
    # Step 1: Initialize Vector Store Manager
    print("🚀 Initializing Vector Store Manager...")
    vs_manager = VectorStoreManager()
    
    # Step 2: Load the vector database (your 95 chunks)
    print("📦 Loading vector database (95 embeddings)...")
    vs_manager.load_vectorstore()
    print("✅ Vector store loaded!\n")
    
    # Step 3: Initialize RAG Chain with the vector store
    print("🔗 Initializing RAG Chain...")
    rag = RAGChain(vs_manager)
    print("✅ RAG Chain ready!\n")
    
    # Step 4: Initialize Cover Letter Generator with RAG chain
    print("📝 Initializing Cover Letter Generator...")
    generator = CoverLetterGenerator(rag)
    print("✅ Generator ready!\n")
    
    # Step 5: Generate cover letter (creates .docx file automatically)
    result = generator.generate_cover_letter(job_description)
    
    # Print final summary
    print(f"\n✨ SUCCESS! Your cover letter is ready!")
    print(f"📄 File location: {result['file_path']}")

if __name__ == "__main__":
    main()