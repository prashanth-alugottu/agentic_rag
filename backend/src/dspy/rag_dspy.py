import dspy

class RAGSignature(dspy.Signature):
    context = dspy.InputField(desc="Relevant retrieved documents")
    question = dspy.InputField(desc="User query")

    reasoning = dspy.OutputField(desc="Step by step reasoning")
    answer = dspy.OutputField(desc="Final grounded answer (max 3 lines)")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")


class MultiStepRAG(dspy.Module):
    def __init__(self):
        self.initial = dspy.ChainOfThought(RAGSignature)
        self.refiner = dspy.ChainOfThought(RAGSignature)

    def forward(self, context, question):
        # Step 1: Initial answer
        first = self.initial(context=context, question=question)

        # ✅ Condition 1: If no answer → return directly
        if not first.answer or "i don't know" in first.answer.lower():
            print("Returning from first because it is i dont know or none ")
            return first

        # ✅ Condition 2: If high confidence → skip refinement
        try:
            conf = float(first.confidence)
            print("confidence score from the first : ",conf)
        except:
            conf = 0.5

        if conf > 0.8:
            print("⚡ Skipping refinement (high confidence)")
            return first

        # Step 2: Refinement ONLY if needed
        print("🔁 Running refinement step")

        refined = self.refiner(
            context=context,
            question=f"""
            Improve this answer:
            {first.answer}

            Make it more accurate, concise and strictly grounded in context.
            """
        )

        return refined
    
class ContextFilter(dspy.Module):
    def __init__(self):
        self.filter = dspy.ChainOfThought(
            "context, question -> filtered_context"
        )

    def forward(self, context, question):
        return self.filter(context=context, question=question)