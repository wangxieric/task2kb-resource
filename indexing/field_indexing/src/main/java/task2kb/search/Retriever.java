/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package task2kb.search;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;

/**
 *
 * @author prochetasen
 */
public class Retriever {
    
     IndexReader reader;
    IndexSearcher searcher;

    public Retriever(String indexPath) throws IOException {
        File indexDir = new File(indexPath);
        reader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity(0.8f, 0.5f));

    }
    
    
      public TopDocs retrieve(String qryString) throws ParseException, IOException, Exception {

        String words[] = qryString.split("\\s+");
        BooleanQuery bq = new BooleanQuery();
        for (String s : words) {
            Term term1 = new Term("title", s);
            //create the term query object
            Query query1 = new TermQuery(term1);
            //query1.setBoost(1.2f);
            bq.add(query1, BooleanClause.Occur.SHOULD);
            
            term1 = new Term("steps", s);
            //create the term query object
            query1 = new TermQuery(term1);
            //query1.setBoost(1.2f);
            bq.add(query1, BooleanClause.Occur.SHOULD);
            
             term1 = new Term("intro", s);
            //create the term query object
            query1 = new TermQuery(term1);
            //query1.setBoost(1.2f);
            bq.add(query1, BooleanClause.Occur.SHOULD);
        }
       
        //System.out.println("Query: " + bq);
        TopDocs tdocs = searcher.search(bq, 1000);
        return tdocs;
    }
     public static void main(String[] args) throws Exception {
       //args[0] index path
       Retriever tr = new Retriever(args[0]);
        //args[1] Query String
        tr.retrieve(args[1]);

    }
}
