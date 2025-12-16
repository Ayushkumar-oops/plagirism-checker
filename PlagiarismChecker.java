package plagiarismchecker;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class PlagiarismChecker {

    /* ===================== DOCUMENT CLASS ===================== */
    static class Document {
        String name;
        Map<String, Integer> termFreq = new HashMap<>();
        int totalTerms = 0;

        Document(String name) {
            this.name = name;
        }

        void tokenize(String text, Set<String> stopwords) {
            termFreq.clear();
            totalTerms = 0;

            Pattern pattern = Pattern.compile("\\p{L}+(?:'\\p{L}+)?");
            Matcher matcher = pattern.matcher(text.toLowerCase(Locale.ROOT));

            while (matcher.find()) {
                String token = matcher.group();
                if (stopwords != null && stopwords.contains(token)) continue;
                termFreq.put(token, termFreq.getOrDefault(token, 0) + 1);
                totalTerms++;
            }
        }

        double tf(String token) {
            if (totalTerms == 0) return 0.0;
            return termFreq.getOrDefault(token, 0) / (double) totalTerms;
        }
    }

    /* ===================== COSINE SIMILARITY ===================== */
    static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0 || normB == 0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /* ===================== FILE LOADER ===================== */
    static List<Path> listTxtFiles(Path dir) throws IOException {
        try (var stream = Files.walk(dir)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                    .collect(Collectors.toList());
        }
    }

    /* ===================== IDF ===================== */
    static Map<String, Double> computeIdf(List<Document> docs, Set<String> vocab) {
        Map<String, Double> idf = new HashMap<>();
        int N = docs.size();

        for (String term : vocab) {
            int df = 0;
            for (Document d : docs) {
                if (d.termFreq.containsKey(term)) df++;
            }
            idf.put(term, Math.log((double) N / (1 + df)) + 1);
        }
        return idf;
    }

    /* ===================== TF-IDF VECTOR ===================== */
    static double[] buildTfIdfVector(Document d, List<String> vocab, Map<String, Double> idf) {
        double[] vec = new double[vocab.size()];
        for (int i = 0; i < vocab.size(); i++) {
            String term = vocab.get(i);
            vec[i] = d.tf(term) * idf.getOrDefault(term, 0.0);
        }
        return vec;
    }

    /* ===================== STOPWORDS ===================== */
    static Set<String> defaultStopwords() {
        return new HashSet<>(Arrays.asList(
                "the","is","in","and","a","an","to","of","that","this","it",
                "for","on","with","as","by","are","from","at","be","or","was","were","which"
        ));
    }

    /* ===================== MAIN ===================== */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        List<Document> documents = new ArrayList<>();
        List<Path> loadedFiles = new ArrayList<>();

        boolean useStopwords = true;
        double threshold = 70.0; // percentage

        while (true) {
            System.out.println("\n========= PLAGIARISM CHECKER =========");
            System.out.println("1. Add directory of .txt files");
            System.out.println("2. Toggle stopwords (Currently " + (useStopwords ? "ON" : "OFF") + ")");
            System.out.println("3. Compute similarity");
            System.out.println("4. Set plagiarism threshold (Current " + threshold + "%)");
            System.out.println("5. Export report (CSV + Flags)");
            System.out.println("6. List loaded files");
            System.out.println("7. Clear loaded files");
            System.out.println("8. Exit");
            System.out.print("Choose option: ");

            int choice;
            try {
                choice = Integer.parseInt(scanner.nextLine());
            } catch (Exception e) {
                System.out.println("Invalid input!");
                continue;
            }

            /* ================= OPTION 1 ================= */
            if (choice == 1) {
                System.out.print("Enter directory path: ");
                Path dir = Paths.get(scanner.nextLine());

                if (!Files.isDirectory(dir)) {
                    System.out.println("Invalid directory!");
                    continue;
                }

                try {
                    documents.clear();
                    loadedFiles.clear();

                    for (Path p : listTxtFiles(dir)) {
                        Document d = new Document(p.getFileName().toString());
                        d.tokenize(Files.readString(p),
                                useStopwords ? defaultStopwords() : null);
                        documents.add(d);
                        loadedFiles.add(p);
                    }
                    System.out.println("Loaded " + documents.size() + " documents.");
                } catch (IOException e) {
                    System.out.println("Error reading files!");
                }
            }

            /* ================= OPTION 2 ================= */
            else if (choice == 2) {
                useStopwords = !useStopwords;
                System.out.println("Stopwords are now " + (useStopwords ? "ON" : "OFF"));
            }

            /* ================= OPTION 3 ================= */
            else if (choice == 3) {
                if (documents.size() < 2) {
                    System.out.println("Load at least 2 documents!");
                    continue;
                }

                Set<String> vocab = new LinkedHashSet<>();
                documents.forEach(d -> vocab.addAll(d.termFreq.keySet()));

                List<String> vocabList = new ArrayList<>(vocab);
                Map<String, Double> idf = computeIdf(documents, vocab);

                DecimalFormat df = new DecimalFormat("#0.00");

                System.out.println("\nSimilarity Matrix (%):");
                System.out.print("Doc\\Doc");
                documents.forEach(d -> System.out.print("\t" + d.name));
                System.out.println();

                for (int i = 0; i < documents.size(); i++) {
                    System.out.print(documents.get(i).name);
                    double[] v1 = buildTfIdfVector(documents.get(i), vocabList, idf);

                    for (int j = 0; j < documents.size(); j++) {
                        double[] v2 = buildTfIdfVector(documents.get(j), vocabList, idf);
                        double sim = cosineSimilarity(v1, v2) * 100;
                        System.out.print("\t" + df.format(sim) + "%");
                    }
                    System.out.println();
                }
            }

            /* ================= OPTION 4 ================= */
            else if (choice == 4) {
                System.out.print("Enter threshold (0–100): ");
                try {
                    double t = Double.parseDouble(scanner.nextLine());
                    if (t < 0 || t > 100) throw new Exception();
                    threshold = t;
                } catch (Exception e) {
                    System.out.println("Invalid threshold!");
                }
            }

            /* ================= OPTION 5 ================= */
            else if (choice == 5) {
                if (documents.size() < 2) {
                    System.out.println("Compute similarity first!");
                    continue;
                }

                try (PrintWriter csv = new PrintWriter("similarity_matrix.csv");
                     PrintWriter flags = new PrintWriter("plagiarism_flags.txt")) {

                    Set<String> vocab = new LinkedHashSet<>();
                    documents.forEach(d -> vocab.addAll(d.termFreq.keySet()));
                    List<String> vocabList = new ArrayList<>(vocab);
                    Map<String, Double> idf = computeIdf(documents, vocab);

                    csv.print("Document");
                    documents.forEach(d -> csv.print("," + d.name));
                    csv.println();

                    for (int i = 0; i < documents.size(); i++) {
                        csv.print(documents.get(i).name);
                        double[] v1 = buildTfIdfVector(documents.get(i), vocabList, idf);

                        for (int j = 0; j < documents.size(); j++) {
                            double[] v2 = buildTfIdfVector(documents.get(j), vocabList, idf);
                            double sim = cosineSimilarity(v1, v2) * 100;
                            csv.print("," + String.format("%.2f", sim));

                            if (i < j && sim >= threshold) {
                                flags.printf("Plagiarism detected: %s & %s (%.2f%%)%n",
                                        documents.get(i).name,
                                        documents.get(j).name,
                                        sim);
                            }
                        }
                        csv.println();
                    }
                    System.out.println("Reports generated successfully!");
                } catch (Exception e) {
                    System.out.println("Error exporting files!");
                }
            }

            /* ================= OPTION 6 ================= */
            else if (choice == 6) {
                loadedFiles.forEach(p -> System.out.println(p.toString()));
            }

            /* ================= OPTION 7 ================= */
            else if (choice == 7) {
                documents.clear();
                loadedFiles.clear();
                System.out.println("Cleared files!");
            }

            /* ================= OPTION 8 ================= */
            else if (choice == 8) {
                System.out.println("Exiting...");
                break;
            }
        }
        scanner.close();
    }
}
