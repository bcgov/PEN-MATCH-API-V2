from typing import Dict, Any, Optional, Type
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class LLMClient:
    """Centralized LLM client using LangChain with reusable methods"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        # Initialize LangChain ChatOpenAI
        llm_kwargs = {
            "api_key": api_key,
            "model": model,
            "temperature": 0.1
        }
        
        if base_url:
            llm_kwargs["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.model = model
        self.base_url = base_url
        
        # Create different temperature variants
        self.creative_llm = ChatOpenAI(**{**llm_kwargs, "temperature": 0.7})
        self.analytical_llm = ChatOpenAI(**{**llm_kwargs, "temperature": 0.1})
    
    def with_structured_output(self, schema_class: Type[BaseModel]):
        """Create structured output handler for Pydantic schemas using LangChain"""
        def invoke(prompt: str, temperature: float = 0.1) -> BaseModel:
            try:
                # Create a chain for structured output
                parser = JsonOutputParser(pydantic_object=schema_class)
                
                # Create prompt template with format instructions
                prompt_template = ChatPromptTemplate.from_messages([
                    HumanMessagePromptTemplate.from_template(
                        "{prompt}\n\n{format_instructions}"
                    )
                ])
                
                # Use appropriate LLM based on temperature
                llm = self.creative_llm if temperature > 0.5 else self.analytical_llm
                llm.temperature = temperature
                
                # Create the chain
                chain = (
                    {
                        "prompt": RunnablePassthrough(),
                        "format_instructions": lambda x: parser.get_format_instructions()
                    }
                    | prompt_template
                    | llm
                    | parser
                )
                
                # Invoke the chain
                result = chain.invoke(prompt)
                
                # Convert to Pydantic model if needed
                if isinstance(result, dict):
                    return schema_class(**result)
                return result
                
            except Exception as e:
                print(f"Error in LangChain LLM call: {e}")
                # Return default "no match" response on error
                return schema_class(
                    decision="NO_MATCH",
                    confidence=0.0,
                    reasons=[f"LLM analysis failed: {str(e)}"],
                    mismatches={}
                )
        
        return type('StructuredOutput', (), {'invoke': invoke})()
    
    def with_structured_output_and_system(self, schema_class: Type[BaseModel]):
        """Create structured output handler with system prompt support for Pydantic schemas using LangChain"""
        def invoke(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> BaseModel:
            try:
                # Create a chain for structured output with system prompt
                parser = JsonOutputParser(pydantic_object=schema_class)
                
                # Create prompt template with system and user messages + format instructions
                prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(
                        user_prompt + "\n\n{format_instructions}"
                    )
                ])
                
                # Use appropriate LLM based on temperature
                llm = self.creative_llm if temperature > 0.5 else self.analytical_llm
                llm.temperature = temperature
                
                # Create the chain with format instructions
                chain = prompt_template | llm | parser
                
                # Invoke the chain with format instructions
                result = chain.invoke({
                    "format_instructions": parser.get_format_instructions()
                })
                
                # Convert to Pydantic model if needed
                if isinstance(result, dict):
                    return schema_class(**result)
                return result
                
            except Exception as e:
                print(f"Error in LangChain LLM call with system prompt: {e}")
                # Return default "no match" response on error
                return schema_class(
                    decision="NO_MATCH",
                    confidence=0.0,
                    reasons=[f"LLM analysis failed: {str(e)}"],
                    mismatches={}
                )
        
        return type('StructuredOutputWithSystem', (), {'invoke': invoke})()
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Simple text generation using LangChain"""
        try:
            # Create LLM with specific settings
            llm = ChatOpenAI(
                api_key=self.llm.api_key,
                model=self.model,
                base_url=self.base_url,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Create simple chain
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            
            result = chain.invoke({"prompt": prompt})
            return result
            
        except Exception as e:
            print(f"Error in LangChain text generation: {e}")
            return f"Error: {str(e)}"
    
    def analyze_with_context(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Analysis with system context using LangChain"""
        try:
            # Create LLM with specific temperature
            llm = ChatOpenAI(
                api_key=self.llm.api_key,
                model=self.model,
                base_url=self.base_url,
                temperature=temperature
            )
            
            # Create prompt template with system and human messages
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(user_prompt)
            ])
            
            # Create chain
            chain = prompt_template | llm | StrOutputParser()
            
            result = chain.invoke({})
            return result
            
        except Exception as e:
            print(f"Error in LangChain contextual analysis: {e}")
            return f"Error: {str(e)}"
    
    def create_custom_chain(self, prompt_template: str, output_parser=None, temperature: float = 0.1):
        """Create a custom LangChain chain for specific use cases"""
        try:
            # Create LLM with specific temperature
            llm = ChatOpenAI(
                api_key=self.llm.api_key,
                model=self.model,
                base_url=self.base_url,
                temperature=temperature
            )
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create chain with or without parser
            if output_parser:
                chain = prompt | llm | output_parser
            else:
                chain = prompt | llm | StrOutputParser()
            
            return chain
            
        except Exception as e:
            print(f"Error creating custom chain: {e}")
            return None
    
    def batch_analyze(self, prompts: list, temperature: float = 0.1) -> list:
        """Batch process multiple prompts efficiently"""
        try:
            # Create LLM for batch processing
            llm = ChatOpenAI(
                api_key=self.llm.api_key,
                model=self.model,
                base_url=self.base_url,
                temperature=temperature
            )
            
            # Create simple chain
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            
            # Use batch processing
            inputs = [{"prompt": prompt} for prompt in prompts]
            results = chain.batch(inputs)
            
            return results
            
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return [f"Error: {str(e)}" for _ in prompts]