import pickle
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
warnings.filterwarnings("ignore")

class RAGSystem:
    def __init__(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model_name="IlyaGusev/saiga_yandexgpt_8b"):
        """
        Инициализация RAG системы
        
        Args:
            embedding_model_name: модель для эмбеддингов
            llm_model_name: языковая модель для генерации ответов
        """
        
        # Инициализация модели для эмбеддингов
        print("Загрузка модели для эмбеддингов...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Инициализация языковой модели
        print("Загрузка языковой модели...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Конфигурация генерации
        self.generation_config = GenerationConfig.from_pretrained(llm_model_name)
        
        self.documents = []
        self.embeddings = None
        
    def load_data(self, data_path):
        """
        Загрузка данных из pickle файла
        
        Args:
            data_path: путь к pickle файлу с данными
        """
        print(f"Загрузка данных из {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Предполагаем, что данные содержат 'documents' и 'questions'
        self.documents = data.get('documents', [])
        self.questions = data.get('questions', [])
        
        print(f"Загружено {len(self.documents)} документов и {len(self.questions)} вопросов")
        
    def create_embeddings(self):
        """Создание эмбеддингов для всех документов"""
        if not self.documents:
            raise ValueError("Документы не загружены. Сначала вызовите load_data()")
            
        print("Создание эмбеддингов для документов...")
        self.embeddings = self.embedding_model.encode(self.documents)
        print(f"Создано {len(self.embeddings)} эмбеддингов")
        
    def retrieve_documents(self, query, top_k=3):
        """
        Поиск релевантных документов для запроса
        
        Args:
            query: текстовый запрос
            top_k: количество возвращаемых документов
            
        Returns:
            list: список релевантных документов
        """
        if self.embeddings is None:
            raise ValueError("Эмбеддинги не созданы. Сначала вызовите create_embeddings()")
            
        # Создание эмбеддинга для запроса
        query_embedding = self.embedding_model.encode([query])
        
        # Вычисление косинусного сходства
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Получение индексов наиболее релевантных документов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Возврат наиболее релевантных документов
        retrieved_docs = []
        for idx in top_indices:
            retrieved_docs.append({
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
            
        return retrieved_docs
    
    def format_prompt(self, question, retrieved_docs):
        """
        Форматирование промпта для языковой модели
        
        Args:
            question: вопрос пользователя
            retrieved_docs: релевантные документы
            
        Returns:
            str: форматированный промпт
        """
        context = "\n".join([f"[{i+1}] {doc['document']}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""<|system|>Ты - полезный AI ассистент. Ответь на вопрос пользователя, используя предоставленную информацию. Если информации недостаточно, скажи об этом.</s>
<|user|>Контекст:
{context}

Вопрос: {question}

Ответ:</s>
<|assistant|>"""
        
        return prompt
    
    def generate_answer(self, prompt, max_length=512, temperature=0.7):
        """
        Генерация ответа с помощью языковой модели
        
        Args:
            prompt: входной промпт
            max_length: максимальная длина ответа
            temperature: температура для генерации
            
        Returns:
            str: сгенерированный ответ
        """
        # Токенизация промпта
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        
        # Генерация ответа
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=self.generation_config
            )
        
        # Декодирование ответа
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение только ответа ассистента
        assistant_start = response.find("<|assistant|>") + len("<|assistant|>")
        answer = response[assistant_start:].strip()
        
        return answer
    
    def answer_question(self, question, top_k=3):
        """
        Полный процесс ответа на вопрос
        
        Args:
            question: вопрос пользователя
            top_k: количество релевантных документов для поиска
            
        Returns:
            dict: ответ и релевантные документы
        """
        print(f"Обработка вопроса: {question}")
        
        # Поиск релевантных документов
        retrieved_docs = self.retrieve_documents(question, top_k=top_k)
        
        # Форматирование промпта
        prompt = self.format_prompt(question, retrieved_docs)
        
        # Генерация ответа
        answer = self.generate_answer(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'prompt': prompt
        }
    
    def evaluate_questions(self, questions=None, top_k=3):
        """
        Оценка системы на множестве вопросов
        
        Args:
            questions: список вопросов (если None, используется self.questions)
            top_k: количество релевантных документов для поиска
            
        Returns:
            list: результаты для всех вопросов
        """
        if questions is None:
            questions = self.questions
            
        results = []
        for i, question in enumerate(questions):
            print(f"Обработка вопроса {i+1}/{len(questions)}: {question}")
            
            try:
                result = self.answer_question(question, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Ошибка при обработке вопроса '{question}': {e}")
                results.append({
                    'question': question,
                    'answer': f"Ошибка: {e}",
                    'retrieved_documents': [],
                    'error': str(e)
                })
                
        return results

def main():
    # Инициализация RAG системы
    rag_system = RAGSystem()
    
    # Загрузка данных (замените на ваш путь к файлу)
    data_path = "your_data.pkl"  # Замените на актуальный путь
    rag_system.load_data(data_path)
    
    # Создание эмбеддингов
    rag_system.create_embeddings()
    
    # Пример ответа на один вопрос
    question = "Ваш вопрос здесь"
    result = rag_system.answer_question(question)
    
    print(f"\nВопрос: {result['question']}")
    print(f"Ответ: {result['answer']}")
    print(f"\nРелевантные документы:")
    for i, doc in enumerate(result['retrieved_documents']):
        print(f"[{i+1}] (сходство: {doc['similarity']:.4f}): {doc['document']}")
    
    # Оценка на всех вопросах
    print("\n" + "="*50)
    print("Оценка на всех вопросах:")
    print("="*50)
    
    all_results = rag_system.evaluate_questions()
    
    # Сохранение результатов
    output_path = "rag_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nРезультаты сохранены в {output_path}")

if __name__ == "__main__":
    main()
